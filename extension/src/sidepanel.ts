// Side panel: which capture path is in use, live landmark stats, the thumbnail strip, and probe results.
import { FEATURES_PER_FRAME, SEQUENCE_LENGTH } from '@shared/features'
import type { LandmarkRecording } from '@shared/layout'
import { send, THUMB_BUFFER_SECONDS, type CaptureStatus, type Msg, type ProbeResult, type Thumb } from './messages'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const strip = $('strip')
const statusEl = $('status')
const pathEl = $('path')
const probeList = $('probe')
const logEl = $('log')
let thumbs: Thumb[] = []

$('shared').textContent = `Shared web code loaded (model input ${SEQUENCE_LENGTH} × ${FEATURES_PER_FRAME}).`

/** Probe results and error strings can come from the page: escape before putting them in markup. */
const esc = (s: unknown) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!)

function log(text: string) {
  logEl.textContent = `${new Date().toLocaleTimeString()} ${text}\n${logEl.textContent}`.slice(0, 4000)
}

function addThumb(t: Thumb) {
  thumbs.push(t)
  const img = document.createElement('img')
  img.src = t.dataUrl
  img.alt = 'Captured frame'
  img.setAttribute('role', 'listitem')
  strip.appendChild(img)
  while (thumbs.length && thumbs[0].t < t.t - THUMB_BUFFER_SECONDS * 1000) {
    thumbs.shift()
    strip.firstElementChild?.remove()
  }
}

function showPath(s: CaptureStatus) {
  pathEl.className = 'path'
  if (!s.path || (!s.capturing && !s.error)) {
    pathEl.textContent = 'Capture path: none yet'
    return
  }
  if (s.path === 'video-element') {
    pathEl.classList.add('primary')
    pathEl.innerHTML = `<strong>Capture path: page &lt;video&gt; element</strong> (primary)<br>
      <span class="muted">${esc(s.reason ?? '')}. Landmarks run on native ${esc(s.sourceWidth ?? '?')} × ${esc(s.sourceHeight ?? '?')} frames.</span>`
  } else {
    pathEl.classList.add('fallback')
    const crop = s.crop
      ? `cropped to the video: ${esc(s.sourceWidth ?? '?')} × ${esc(s.sourceHeight ?? '?')} of the ${esc(s.tabWidth ?? '?')} × ${esc(s.tabHeight ?? '?')} tab`
      : `full tab ${esc(s.tabWidth ?? '?')} × ${esc(s.tabHeight ?? '?')} (no video rect to crop to)`
    pathEl.innerHTML = `<strong>Capture path: tab capture</strong> (fallback)<br>
      <span class="muted">Why: ${esc(s.reason ?? 'unknown')}. ${crop}.</span>`
  }
}

function showStatus(s: CaptureStatus) {
  showPath(s)
  if (s.error) {
    statusEl.innerHTML = `<span class="bad">Error:</span> ${esc(s.error)}`
    return
  }
  if (!s.capturing) {
    statusEl.textContent = s.path ? 'Starting…' : 'Not capturing. Press Alt+Shift+S on the video tab.'
    return
  }
  const d = s.lastDetected
  const yes = (b?: boolean) => (b ? '<span class="ok">yes</span>' : 'no')
  statusEl.innerHTML =
    `<span class="ok">Capturing</span> · MediaPipe on ${esc(s.delegate ?? '?')}<br>` +
    `landmarks ${esc(s.landmarkFps ?? '–')} fps · thumbnails ${esc(s.thumbFps ?? '–')} fps<br>` +
    `landmark buffer ${esc(s.bufferFrames ?? 0)} frames, ${esc(s.bufferSeconds ?? 0)} s<br>` +
    (d ? `last frame: left hand ${yes(d.left)} · right hand ${yes(d.right)} · face ${yes(d.face)} · pose ${yes(d.pose)}` : '')
}

function showProbe(r: ProbeResult) {
  if (probeList.firstElementChild?.classList.contains('muted')) probeList.innerHTML = ''
  const li = document.createElement('li')
  const verdict =
    r.tainted === true
      ? '<span class="bad">canvas tainted</span>: cannot read the video directly'
      : r.tainted === false
        ? r.blank
          ? '<span class="bad">readable but blank</span> (likely DRM-protected video)'
          : '<span class="ok">canvas readable</span>'
        : 'no drawable &lt;video&gt; found'
  li.innerHTML = `<strong>${r.isTopFrame ? 'Top frame' : 'Frame'}</strong> ${esc(r.frameUrl)}<br>${esc(r.videos)} video element(s). ${verdict}`
  if (r.chosen) li.innerHTML += `<br><span class="muted">${esc(r.chosen.width)} × ${esc(r.chosen.height)}, ${r.chosen.paused ? 'paused' : 'playing'}</span>`
  if (r.error) li.innerHTML += `<br><span class="muted">${esc(r.error)}</span>`
  if (r.thumb?.startsWith('data:image/')) {
    const img = document.createElement('img')
    img.src = r.thumb
    img.alt = 'Frame read through the page video canvas'
    li.appendChild(img)
  }
  probeList.prepend(li)
}

chrome.runtime.onMessage.addListener((msg: Msg) => {
  if (msg.target !== 'panel') return
  if (msg.type === 'thumb') addThumb(msg.thumb)
  else if (msg.type === 'status') showStatus(msg.status)
  else if (msg.type === 'probe-result') showProbe(msg.result)
  else if (msg.type === 'log') log(msg.text)
})

$('stop').addEventListener('click', () => void send({ target: 'sw', type: 'stop' }))
$('probe-btn').addEventListener('click', () => void send({ target: 'sw', type: 'probe' }))
$('download').addEventListener('click', async () => {
  const rec = await send<LandmarkRecording | null>({ target: 'pipeline', type: 'get-landmarks' })
  if (!rec) {
    log('no landmarks buffered yet')
    return
  }
  const url = URL.createObjectURL(new Blob([JSON.stringify(rec)], { type: 'application/json' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `landmarks-extension-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  a.click()
  URL.revokeObjectURL(url)
  log(`saved ${rec.frames.length} landmark frames (${rec.width} × ${rec.height}); put it in demo/landmarks/ for the parity test`)
})

// If capture is already running (panel reopened), load the current buffer.
void send<{ thumbs: Thumb[]; status: CaptureStatus }>({ target: 'pipeline', type: 'get-buffer' }).then((r) => {
  if (!r) return
  thumbs = []
  strip.innerHTML = ''
  r.thumbs.forEach(addThumb)
  showStatus(r.status)
})
