// MV3 service worker. On Alt+Shift+S (or the toolbar button opens the panel; the hotkey starts capture):
//  1. open the side panel (must happen synchronously inside the hotkey gesture)
//  2. probe every frame: can the page's <video> be drawn to a readable canvas?
//  3. PRIMARY: readable and not blank -> capture-video.js in that frame draws the <video> at native resolution and
//     streams frames to a hidden extension iframe that runs landmarks.
//     FALLBACK: tainted / blank / no video / primary failed -> tab capture in the offscreen document, cropped to the
//     video's on-screen rect.
import { send, type Crop, type Msg, type ProbeResult, type Viewport, type CsMsg } from './messages'

chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(console.error)

let activeTab: number | null = null
let activeFrame: number | null = null

const log = (text: string) => {
  console.log('[SignSeek]', text)
  void send({ target: 'panel', type: 'log', text })
}

const toTab = (tabId: number, msg: CsMsg, frameId?: number) =>
  chrome.tabs.sendMessage(tabId, msg, frameId === undefined ? undefined : { frameId }).catch(() => {})

async function ensureOffscreen(): Promise<void> {
  const existing = await chrome.runtime.getContexts({ contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT] })
  if (existing.length > 0) return
  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: [chrome.offscreen.Reason.USER_MEDIA],
    justification: 'Fallback capture: read frames from the captured tab when the page video cannot be read directly.',
  })
}

/** The offscreen listener may not be registered the instant createDocument resolves; retry briefly. */
async function toOffscreen(msg: Msg): Promise<void> {
  for (let i = 0; i < 20; i++) {
    try {
      await chrome.runtime.sendMessage(msg)
      return
    } catch {
      await new Promise((r) => setTimeout(r, 50))
    }
  }
  throw new Error('offscreen document did not respond')
}

async function stopAll(): Promise<void> {
  await send({ target: 'pipeline', type: 'stop' }) // whichever pipeline is active (iframe or offscreen)
  if (activeTab !== null) await toTab(activeTab, { cs: 'stop' }) // content scripts in all frames
  activeTab = null
  activeFrame = null
}

async function probe(tabId: number): Promise<{ frameId: number; result: ProbeResult }[]> {
  try {
    const res = await chrome.scripting.executeScript({ target: { tabId, allFrames: true }, files: ['probe.js'] })
    const out = res
      .filter((r) => r.result)
      .map((r) => ({ frameId: r.frameId, result: r.result as ProbeResult }))
    for (const r of out) void send({ target: 'panel', type: 'probe-result', result: r.result })
    return out
  } catch (e) {
    log(`probe could not run on this page: ${(e as Error).message}`)
    return []
  }
}

function cropFrom(r: ProbeResult | undefined): Crop | null {
  if (!r?.rect) return null
  const { width: vw, height: vh } = r.viewport
  const x0 = Math.max(0, r.rect.x) / vw
  const y0 = Math.max(0, r.rect.y) / vh
  const x1 = Math.min(vw, r.rect.x + r.rect.width) / vw
  const y1 = Math.min(vh, r.rect.y + r.rect.height) / vh
  return x1 > x0 && y1 > y0 ? { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } : null
}

async function startVideoPath(tabId: number, frameId: number): Promise<void> {
  const nonce = crypto.randomUUID()
  await chrome.storage.session.set({ captureNonce: nonce }) // readable by the capture iframe, not by the page
  await chrome.scripting.executeScript({ target: { tabId, frameIds: [frameId] }, files: ['capture-video.js'] })
  await toTab(tabId, { cs: 'start-video', nonce, frameUrl: chrome.runtime.getURL('capture-frame.html') }, frameId)
  activeTab = tabId
  activeFrame = frameId
  log(`primary path: page <video> element (frame ${frameId})`)
}

async function startTabPath(tabId: number, reason: string, top?: ProbeResult): Promise<void> {
  await ensureOffscreen()
  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId })
  const crop = cropFrom(top) // null -> full tab
  const viewport: Viewport | null = top?.viewport ?? null
  await toOffscreen({ target: 'offscreen', type: 'start', streamId, crop, viewport, reason })
  activeTab = tabId
  // keep the crop rect current while the user scrolls/resizes (top-frame videos only)
  if (top?.rect) {
    await chrome.scripting.executeScript({ target: { tabId, frameIds: [0] }, files: ['capture-video.js'] })
    await toTab(tabId, { cs: 'track-rect' }, 0)
  }
  log(`fallback path: tab capture${crop ? ', cropped to the video rect' : ' (full tab, no top-frame video to crop to)'} (${reason})`)
}

async function begin(tabId: number): Promise<void> {
  await stopAll()
  const results = await probe(tabId)
  const withVideo = results.filter((r) => r.result.chosen)
  const area = (r: ProbeResult) => (r.rect ? r.rect.width * r.rect.height : 0)
  const readable = withVideo
    .filter((r) => r.result.tainted === false && !r.result.blank)
    .sort((a, b) => area(b.result) - area(a.result))[0]
  if (readable) {
    try {
      await startVideoPath(tabId, readable.frameId)
      return
    } catch (e) {
      log(`primary path could not start: ${(e as Error).message}`)
    }
  }
  const top = results.find((r) => r.frameId === 0)?.result
  const best = withVideo.sort((a, b) => area(b.result) - area(a.result))[0]?.result
  const reason = readable
    ? 'page video path failed to start'
    : !best
      ? 'no playing <video> found'
      : best.tainted
        ? 'page video canvas is tainted'
        : best.blank
          ? 'page video canvas is blank (likely DRM)'
          : 'page video could not be read'
  if (best && !best.isTopFrame) log('the video is inside an iframe: tab capture will use the full tab (crop needs a top-frame video)')
  await startTabPath(tabId, reason, top?.chosen ? top : undefined)
}

chrome.commands.onCommand.addListener((command, tab) => {
  if (command !== 'lookup-sign' || tab?.id === undefined) return
  // Must be called synchronously inside the hotkey's user gesture, before any await.
  chrome.sidePanel.open({ windowId: tab.windowId }).catch(console.error)
  begin(tab.id).catch((e) => log(`capture failed: ${(e as Error).message}`))
})

chrome.runtime.onMessage.addListener((msg: Msg, sender) => {
  if (msg.target !== 'sw') return
  if (msg.type === 'probe') {
    chrome.tabs.query({ active: true, lastFocusedWindow: true }).then(([tab]) => {
      if (tab?.id !== undefined) void probe(tab.id)
    })
  } else if (msg.type === 'stop') {
    void stopAll()
  } else if (msg.type === 'video-path-failed') {
    const tabId = sender.tab?.id
    if (tabId === undefined || tabId !== activeTab || sender.frameId !== activeFrame) return
    log(`primary path stopped: ${msg.reason}; switching to tab capture`)
    void send({ target: 'pipeline', type: 'stop' }).then(() =>
      probe(tabId).then((results) => {
        const top = results.find((r) => r.frameId === 0)?.result
        return startTabPath(tabId, msg.reason, top?.chosen ? top : undefined)
      }),
    ).catch((e) => log(`fallback failed: ${(e as Error).message}`))
  }
})
