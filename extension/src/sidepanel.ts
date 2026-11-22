// Side panel: look up the last 1-5 s of the captured video (button or Alt+Shift+S pressed again), shown with the
// web app's result cards and dictionary.json; capture diagnostics (path, landmark stats, probe, log) below.
import '@shared/app.css'
import { getModel, loadReferenceData } from '@shared/app/engine'
import { renderCards } from '@shared/app/results'
import { frameFromNested, type LandmarkRecording } from '@shared/layout'
import { toTrainingAspect } from '@shared/landmarks'
import { lookupSign } from '@shared/lookup'
import { send, THUMB_BUFFER_SECONDS, type CaptureStatus, type Msg, type ProbeResult, type Thumb } from './messages'

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const TTA_VARIANTS = 5
const SECONDS_KEY = 'signdex.lookbackSeconds'

const slider = $<HTMLInputElement>('seconds')
const lookupBtn = $<HTMLButtonElement>('lookup')
const lookupStatus = $('lookup-status')
const strip = $('strip')
let thumbs: Thumb[] = []
let capturing = false
let busy = false

/** Probe results and error strings can come from the page: escape before putting them in markup. */
const esc = (s: unknown) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!)

// ---------------------------------------------------------------- how far back (1-5 s), remembered
try {
  const saved = Number(localStorage.getItem(SECONDS_KEY))
  if (saved >= 1 && saved <= 5) slider.value = String(saved)
} catch {
  /* storage unavailable: default 3 s */
}
const seconds = () => Number(slider.value)
function onSeconds() {
  $('seconds-out').textContent = `${seconds()} s`
  $('lookup-label').textContent = `Look up last ${seconds()} s`
  slider.setAttribute('aria-valuetext', `${seconds()} seconds`)
  drawStrip()
  try {
    localStorage.setItem(SECONDS_KEY, slider.value)
  } catch {
    /* ignore */
  }
}
slider.addEventListener('input', onSeconds)

// ---------------------------------------------------------------- thumbnail strip, last N s highlighted
let stripQueued = false
function drawStrip() {
  if (stripQueued) return
  stripQueued = true
  setTimeout(() => {
    stripQueued = false
    const n = 10 // 2 per second over the 5 s buffer
    strip.replaceChildren()
    if (!thumbs.length) return
    const last = thumbs[thumbs.length - 1].t
    for (let i = 0; i < n; i++) {
      const at = last - THUMB_BUFFER_SECONDS * 1000 + ((i + 1) * THUMB_BUFFER_SECONDS * 1000) / n
      const th = thumbs.reduce((best, x) => (Math.abs(x.t - at) < Math.abs(best.t - at) ? x : best))
      const img = document.createElement('img')
      img.src = th.dataUrl
      img.alt = ''
      if (at > last - seconds() * 1000) img.className = 'in'
      strip.append(img)
    }
  }, 250)
}

function addThumb(t: Thumb) {
  thumbs.push(t)
  while (thumbs.length && thumbs[0].t < t.t - THUMB_BUFFER_SECONDS * 1000) thumbs.shift()
  drawStrip()
}

// ---------------------------------------------------------------- lookup
function updateButton() {
  lookupBtn.disabled = busy || !capturing
}

async function lookup() {
  if (busy) return
  if (!capturing) {
    lookupStatus.textContent = 'Not capturing yet. Press Alt+Shift+S on the tab with the video.'
    return
  }
  busy = true
  updateButton()
  const n = seconds()
  try {
    lookupStatus.textContent = 'Loading the sign model…'
    const [model] = await Promise.all([getModel(), loadReferenceData()])
    lookupStatus.textContent = 'Finding matches…'
    const rec = await send<LandmarkRecording | null>({ target: 'pipeline', type: 'get-landmarks' })
    if (!rec || rec.frames.length === 0) throw new Error('nothing captured yet. Play the video, then try again')
    const lastT = rec.t[rec.t.length - 1]
    const frames = rec.frames.filter((_, i) => rec.t[i] >= lastT - n).map(frameFromNested)
    // same pipeline as the web app: aspect correction to the training video, then features at the model's L
    const corrected = toTrainingAspect(frames, rec.width, rec.height)
    const res = await lookupSign(model, corrected, { variants: TTA_VARIANTS, k: 5 })
    if (res.handFrames === 0) {
      lookupStatus.textContent = `No hands were seen in the last ${n} s. Look up right after the sign, while the signer is in view.`
      return
    }
    renderCards($('cards'), res.matches, 'Not sure. Try a shorter time, just around the sign.')
    $('results').hidden = false
    lookupStatus.textContent = `Looked at ${frames.length} frames (${res.handFrames} with hands) in ${Math.round(res.timing.totalMs)} ms.`
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' })
  } catch (e) {
    lookupStatus.textContent = `Lookup failed: ${(e as Error).message}`
  } finally {
    busy = false
    updateButton()
  }
}
lookupBtn.addEventListener('click', () => void lookup())

// ---------------------------------------------------------------- capture state and diagnostics
const logEl = $('log')
function log(text: string) {
  logEl.textContent = `${new Date().toLocaleTimeString()} ${text}\n${logEl.textContent}`.slice(0, 4000)
}

function showState(s: CaptureStatus) {
  capturing = s.capturing
  updateButton()
  const state = $('state')
  const hands = s.lastDetected && (s.lastDetected.left || s.lastDetected.right)
  state.className = `sp-state ${s.error ? 'bad' : s.capturing ? (hands ? 'live' : 'idle') : ''}`
  $('state-text').textContent = s.error ? 'Error' : s.capturing ? (hands ? 'Hands in view' : 'Capturing') : s.path ? 'Starting…' : 'Not capturing'
  $('intro').hidden = s.capturing
}

function showPath(s: CaptureStatus) {
  const pathEl = $('path')
  pathEl.className = 'path'
  if (!s.path || (!s.capturing && !s.error)) {
    pathEl.textContent = 'Capture path: none yet'
    return
  }
  if (s.path === 'video-element') {
    pathEl.classList.add('primary')
    pathEl.innerHTML = `<strong>Page &lt;video&gt; element</strong> (primary)<br>
      <span class="muted">${esc(s.reason ?? '')}. Landmarks on native ${esc(s.sourceWidth ?? '?')} × ${esc(s.sourceHeight ?? '?')} frames.</span>`
  } else {
    pathEl.classList.add('fallback')
    const crop = s.crop
      ? `cropped to the video: ${esc(s.sourceWidth ?? '?')} × ${esc(s.sourceHeight ?? '?')} of the ${esc(s.tabWidth ?? '?')} × ${esc(s.tabHeight ?? '?')} tab`
      : `full tab ${esc(s.tabWidth ?? '?')} × ${esc(s.tabHeight ?? '?')} (no video rect to crop to)`
    pathEl.innerHTML = `<strong>Tab capture</strong> (fallback)<br><span class="muted">Why: ${esc(s.reason ?? 'unknown')}. ${crop}.</span>`
  }
}

function showStatus(s: CaptureStatus) {
  showState(s)
  showPath(s)
  const statusEl = $('status')
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
    `buffer ${esc(s.bufferFrames ?? 0)} frames, ${esc(s.bufferSeconds ?? 0)} s<br>` +
    (d ? `last frame: left hand ${yes(d.left)} · right hand ${yes(d.right)} · face ${yes(d.face)} · pose ${yes(d.pose)}` : '')
}

function showProbe(r: ProbeResult) {
  const probeList = $('probe')
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

chrome.runtime.onMessage.addListener((msg: Msg, _sender, sendResponse) => {
  if (msg.target !== 'panel') return
  if (msg.type === 'thumb') addThumb(msg.thumb)
  else if (msg.type === 'status') showStatus(msg.status)
  else if (msg.type === 'probe-result') showProbe(msg.result)
  else if (msg.type === 'log') log(msg.text)
  else if (msg.type === 'lookup') {
    sendResponse(true) // tells the service worker the panel is listening
    void lookup()
  }
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
  log(`saved ${rec.frames.length} landmark frames (${rec.width} × ${rec.height})`)
})

onSeconds()
// If capture is already running (panel reopened), load the current buffer and state.
void send<{ thumbs: Thumb[]; status: CaptureStatus }>({ target: 'pipeline', type: 'get-buffer' }).then((r) => {
  if (!r) return
  thumbs = []
  r.thumbs.forEach(addThumb)
  showStatus(r.status)
})
// Warm up the model while the user finds the sign, so the first lookup is quick.
void getModel().catch(() => {})
