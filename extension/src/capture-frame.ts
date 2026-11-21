// Hidden extension iframe injected into the page by capture-video.ts (PRIMARY path).
// Receives native-resolution ImageBitmaps of the page's <video> over a private MessagePort and runs the
// landmark pipeline here, under the extension's CSP (wasm allowed), not in the page.
import { type Msg, type PortMsg } from './messages'
import { LandmarkPipeline } from './pipeline'

let pipeline: LandmarkPipeline | null = null
let port: MessagePort | null = null

window.addEventListener('message', async (e: MessageEvent) => {
  if (e.source !== window.parent || e.data?.type !== 'signseek-init' || port) return
  // Only accept the port if the nonce matches the one the service worker stored (pages can't read storage.session).
  const { captureNonce } = await chrome.storage.session.get('captureNonce')
  if (!captureNonce || e.data.nonce !== captureNonce || !e.ports[0]) return
  await chrome.storage.session.remove('captureNonce') // one-time
  port = e.ports[0]
  pipeline = new LandmarkPipeline('video-element', 'page <video> canvas is readable')
  try {
    await pipeline.init()
  } catch (err) {
    port.postMessage({ type: 'error', error: `MediaPipe failed to load: ${(err as Error).message}` } satisfies PortMsg)
    return
  }
  port.onmessage = async (m: MessageEvent<PortMsg>) => {
    const d = m.data
    if (d.type === 'frame') {
      try {
        const p = pipeline?.push(d.bitmap, d.width, d.height)
        d.bitmap.close()
        await p
      } finally {
        port?.postMessage({ type: 'done' } satisfies PortMsg)
      }
    } else if (d.type === 'stop') {
      pipeline?.stop()
      pipeline = null
    }
  }
  port.postMessage({ type: 'ready' } satisfies PortMsg)
})

chrome.runtime.onMessage.addListener((msg: Msg, _sender, sendResponse) => {
  if (msg.target !== 'pipeline' || !pipeline) return // only the active pipeline answers
  if (msg.type === 'stop') {
    pipeline.stop()
    pipeline = null
    port?.postMessage({ type: 'stop' } satisfies PortMsg) // content script removes this iframe
    sendResponse(true)
  } else if (msg.type === 'get-buffer') {
    sendResponse(pipeline.buffer())
  } else if (msg.type === 'get-landmarks') {
    sendResponse(pipeline.landmarks())
  }
})
