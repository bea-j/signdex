// Trim screen for an uploaded clip: thumbnail strip with a draggable 1-3 s window (mouse, touch or arrow keys),
// the window playing on a loop, and the lookup on exactly that window (frame-accurate extraction at 20 fps).
import { extractClip, toTrainingAspect, type TimedFrame } from '../landmarks'
import { frameAt, LandmarkOverlay } from '../ui/overlay'
import { prefs } from '../ui/prefs'
import { frameStats, lookupStats, partStats, renderDetails } from './details'
import { lookupSign, type LookupResult } from '../lookup'
import { getExtractor, getModel, loadReferenceData } from './engine'
import { renderCards } from './results'
import type { SignMatch } from '../model'
import { onEnter, onLeave, show } from './router'

const MIN_S = 1
const MAX_S = 3
const DEFAULT_S = 2
const FPS = 20
const TTA_VARIANTS = 5

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const fileInput = $<HTMLInputElement>('file-input')
const video = $<HTMLVideoElement>('trim-video')
const strip = $('strip')
const win = $('window')
const hStart = $('handle-start')
const hEnd = $('handle-end')
const playhead = $('playhead')
const lookupBtn = $<HTMLButtonElement>('trim-lookup')
const status = $('trim-status')
const busyEl = $('trim-busy')

const overlay = new LandmarkOverlay(video)
/** Landmarks of the last looked-up window (t = seconds in the file), drawn while it loops. */
let clipFrames: TimedFrame[] = []
let lastLookup: { res: LookupResult; modelLength: number } | null = null
let delegate = ''
let fileName = ''
let url: string | null = null
let duration = 0
let start = 0
let end = 0
let busy = false
let ready = false // model + MediaPipe loaded
let looping = false

const fmt = (t: number, tenths = false) => {
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return `${String(m).padStart(2, '0')}:${tenths ? s.toFixed(1).padStart(4, '0') : String(Math.floor(s)).padStart(2, '0')}`
}
const minLen = () => Math.min(MIN_S, duration)
const maxLen = () => Math.min(MAX_S, duration)

// ---------------------------------------------------------------- choosing a file
export function chooseFile() {
  fileInput.value = ''
  fileInput.click()
}
$('home-upload').addEventListener('click', chooseFile)
$('another-clip').addEventListener('click', chooseFile)

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0]
  if (!file) return
  if (url) URL.revokeObjectURL(url)
  url = URL.createObjectURL(file)
  fileName = file.name
  lastLookup = null
  $('file-name').textContent = file.name
  $('file-name').title = file.name
  duration = 0
  status.textContent = 'Opening the clip…'
  lookupBtn.disabled = true
  strip.querySelectorAll('.thumb').forEach((t) => t.remove())
  clearResults()
  clipFrames = []
  overlay.clear()
  video.src = url
  show('trim')
})

/** Some WebM files (e.g. from MediaRecorder) report Infinity until seeked to the end. */
async function resolveDuration(): Promise<number> {
  if (video.duration !== Infinity) return video.duration
  await new Promise<void>((r) => {
    video.addEventListener('durationchange', () => r(), { once: true })
    video.currentTime = 1e9
  })
  return video.duration
}

video.addEventListener('loadedmetadata', async () => {
  const src = video.src
  const d = await resolveDuration()
  if (video.src !== src) return
  if (!Number.isFinite(d) || d <= 0) {
    status.textContent = 'Could not read the length of this video. Try another file.'
    return
  }
  duration = video.duration
  start = 0
  end = Math.min(DEFAULT_S, duration)
  $('clip-length').textContent = fmt(duration, true)
  layout()
  void thumbnails()
  seekToStart()
  status.textContent = ready ? '' : 'Loading hand tracking and the sign model…'
  updateButton()
  updateDetails()
})
video.addEventListener('error', () => {
  if (video.getAttribute('src')) status.textContent = 'This browser cannot play that video file. Try an MP4 or WebM clip.'
})

// ---------------------------------------------------------------- thumbnails
async function thumbnails() {
  const src = url!
  const n = Math.max(5, Math.min(12, Math.round(strip.clientWidth / 72)))
  const cells: HTMLCanvasElement[] = []
  for (let i = 0; i < n; i++) {
    const cell = document.createElement('div')
    cell.className = 'thumb'
    const c = document.createElement('canvas')
    c.width = 160
    c.height = 90
    cell.append(c)
    strip.insertBefore(cell, win)
    cells.push(c)
  }
  // a second, hidden <video> so thumbnailing doesn't disturb the preview
  const v = document.createElement('video')
  v.muted = true
  v.preload = 'auto'
  v.src = src
  await new Promise((r) => v.addEventListener('loadeddata', r, { once: true }))
  for (let i = 0; i < n; i++) {
    if (url !== src) return // another file was chosen meanwhile
    await new Promise<void>((r) => {
      v.addEventListener('seeked', () => r(), { once: true })
      v.currentTime = Math.min(duration - 0.01, ((i + 0.5) * duration) / n)
    })
    const ctx = cells[i].getContext('2d')!
    // cover-crop into 16:9
    const s = Math.max(160 / v.videoWidth, 90 / v.videoHeight)
    const w = v.videoWidth * s
    const h = v.videoHeight * s
    ctx.drawImage(v, (160 - w) / 2, (90 - h) / 2, w, h)
  }
  v.removeAttribute('src')
  v.load()
}

// ---------------------------------------------------------------- window geometry
function layout() {
  if (!duration) return
  const pad = 4
  const w = strip.clientWidth - 2 * pad
  win.style.left = `${pad + (start / duration) * w}px`
  win.style.width = `${((end - start) / duration) * w}px`
  const len = end - start
  $('window-length').textContent = `${len.toFixed(1)} s`
  $('window-start').textContent = `${fmt(start, true)} – ${fmt(end, true)}`
  for (const [h, value, label] of [
    [hStart, start, 'Start'],
    [hEnd, end, 'End'],
  ] as const) {
    h.setAttribute('aria-valuemin', '0')
    h.setAttribute('aria-valuemax', duration.toFixed(1))
    h.setAttribute('aria-valuenow', value.toFixed(1))
    h.setAttribute('aria-valuetext', `${label} ${value.toFixed(1)} seconds; window ${len.toFixed(1)} seconds`)
  }
}
new ResizeObserver(layout).observe(strip)

function setWindow(s: number, e: number, keep: 'start' | 'end' | 'move') {
  const lo = minLen()
  const hi = maxLen()
  if (keep === 'move') {
    const len = e - s
    s = Math.max(0, Math.min(duration - len, s))
    e = s + len
  } else if (keep === 'start') {
    // the end moved
    e = Math.max(s + lo, Math.min(s + hi, Math.min(duration, e)))
  } else {
    s = Math.min(e - lo, Math.max(e - hi, Math.max(0, s)))
  }
  start = Math.max(0, s)
  end = Math.min(duration, e)
  layout()
  seekToStart()
}

function seekToStart() {
  if (!duration || busy) return
  video.currentTime = start
  void video.play().catch(() => {})
}

// drag: the window body moves it, the handles resize it
let drag: { kind: 'move' | 'start' | 'end'; x0: number; s0: number; e0: number } | null = null
strip.addEventListener('pointerdown', (ev) => {
  if (!duration || busy) return
  const t = ev.target as HTMLElement
  const kind = t === hStart ? 'start' : t === hEnd ? 'end' : t === win ? 'move' : null
  if (!kind) {
    // click on the strip: center the window there
    const b = strip.getBoundingClientRect()
    const at = ((ev.clientX - b.left - 4) / (b.width - 8)) * duration
    const len = end - start
    setWindow(at - len / 2, at + len / 2, 'move')
    return
  }
  drag = { kind, x0: ev.clientX, s0: start, e0: end }
  strip.setPointerCapture(ev.pointerId)
  ev.preventDefault()
})
strip.addEventListener('pointermove', (ev) => {
  if (!drag) return
  const dt = ((ev.clientX - drag.x0) / (strip.clientWidth - 8)) * duration
  if (drag.kind === 'move') setWindow(drag.s0 + dt, drag.e0 + dt, 'move')
  else if (drag.kind === 'start') setWindow(drag.s0 + dt, drag.e0, 'end')
  else setWindow(drag.s0, drag.e0 + dt, 'start')
})
const endDrag = () => (drag = null)
strip.addEventListener('pointerup', endDrag)
strip.addEventListener('pointercancel', endDrag)

// keyboard: arrows move the window (when it has focus) or one edge (when a handle has focus); Shift = bigger steps
function onKey(kind: 'move' | 'start' | 'end') {
  return (ev: KeyboardEvent) => {
    const dir = ev.key === 'ArrowLeft' || ev.key === 'ArrowDown' ? -1 : ev.key === 'ArrowRight' || ev.key === 'ArrowUp' ? 1 : 0
    if (!dir || !duration) return
    ev.preventDefault()
    ev.stopPropagation()
    const d = dir * (ev.shiftKey ? 0.5 : 0.1)
    if (kind === 'move') setWindow(start + d, end + d, 'move')
    else if (kind === 'start') setWindow(start + d, end, 'end')
    else setWindow(start, end + d, 'start')
  }
}
win.addEventListener('keydown', (ev) => ev.target === win && onKey('move')(ev))
hStart.addEventListener('keydown', onKey('start'))
hEnd.addEventListener('keydown', onKey('end'))

// preview: play the window on a loop, with a playhead on the strip
function loop() {
  if (!looping) return
  if (duration && !busy) {
    if (video.currentTime >= end || video.currentTime < start - 0.05) video.currentTime = start
    playhead.style.left = `${4 + (video.currentTime / duration) * (strip.clientWidth - 8)}px`
    overlay.draw(frameAt(clipFrames, video.currentTime)?.frame ?? null, { points: prefs.get('points'), face: prefs.get('face') })
  }
  requestAnimationFrame(loop)
}
video.addEventListener('ended', seekToStart)

// ---------------------------------------------------------------- lookup
function updateButton() {
  lookupBtn.disabled = busy || !ready || !duration
}

async function lookup() {
  if (busy || !ready || !duration) return
  busy = true
  updateButton()
  busyEl.hidden = false
  busyEl.textContent = 'Tracking hands…'
  status.textContent = ''
  try {
    const [extractor, model] = await Promise.all([getExtractor(), getModel()])
    delegate = extractor.delegate
    const frames = await extractClip(extractor, video, start, end, FPS, (done, total) => {
      busyEl.textContent = `Tracking hands… ${Math.round((100 * done) / total)}%`
    })
    clipFrames = frames
    busyEl.textContent = 'Finding matches…'
    const corrected = toTrainingAspect(
      frames.map((f) => f.frame),
      video.videoWidth,
      video.videoHeight,
    )
    const res = await lookupSign(model, corrected, { variants: TTA_VARIANTS, k: 5 })
    if (res.handFrames === 0) {
      status.textContent = 'No hands were found in this part of the clip. Move the box to where the sign is.'
      return
    }
    lastLookup = { res, modelLength: model.length }
    showResults(res.matches)
    updateDetails()
  } catch (e) {
    status.textContent = `Lookup failed: ${(e as Error).message}`
  } finally {
    busy = false
    busyEl.hidden = true
    updateButton()
    if (!$('screen-trim').hidden) seekToStart()
  }
}
lookupBtn.addEventListener('click', () => void lookup())

// ---------------------------------------------------------------- details panel
const detailsEl = $('trim-details')
function updateDetails() {
  detailsEl.hidden = !prefs.get('details')
  if (detailsEl.hidden) return
  renderDetails(detailsEl, [
    {
      title: 'Clip',
      stats: [
        { label: 'File', value: fileName || '–' },
        { label: 'Resolution', value: video.videoWidth ? `${video.videoWidth} × ${video.videoHeight}, ${duration.toFixed(1)} s` : '–' },
        { label: 'Hand tracking', value: delegate ? `MediaPipe on ${delegate}` : 'loads with the first lookup' },
        { label: 'Landmark sampling', value: `${FPS} fps, frame-accurate seeks` },
        ...frameStats(video.videoWidth, video.videoHeight),
      ],
    },
    { title: 'Looked-up window', stats: partStats(clipFrames.map((f) => f.frame), false) },
    { title: 'Last lookup', stats: lastLookup ? lookupStats(clipFrames.length, lastLookup.res, lastLookup.modelLength) : [] },
  ])
}
prefs.onChange(updateDetails)

/** Results panel next to the player: the top 5 for the selected window, updated in place. */
function showResults(matches: SignMatch[]) {
  $('trim-results-empty').hidden = true
  $('trim-note').hidden = false
  const cap = $('trim-caption')
  cap.hidden = false
  cap.textContent = `Matches for ${fmt(start, true)} – ${fmt(end, true)} (${(end - start).toFixed(1)} s), playing on a loop in the player`
  renderCards($('trim-cards'), matches, 'Not sure. Try trimming tighter around the sign.')
  if (matchMedia('(max-width: 1023px)').matches) $('trim-results-title').scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function clearResults() {
  $('trim-cards').replaceChildren()
  $('trim-results-empty').hidden = false
  $('trim-note').hidden = true
  $('trim-caption').hidden = true
}

onEnter('trim', () => {
  looping = true
  requestAnimationFrame(loop)
  if (duration) {
    layout()
    seekToStart()
  }
  Promise.all([getExtractor(), getModel(), loadReferenceData()])
    .then(() => {
      ready = true
      if (status.textContent?.startsWith('Loading')) status.textContent = ''
      updateButton()
    })
    .catch((e) => (status.textContent = `Could not load the sign model or hand tracking: ${(e as Error).message}`))
})
onLeave('trim', () => {
  looping = false
  video.pause()
})

// Dev-server-only test hook (removed from production builds), see webcam.ts.
if (import.meta.env.DEV) {
  Object.assign(window, { __signdexTrim: { setClipFrames: (f: TimedFrame[]) => (clipFrames = f), window: () => [start, end] } })
}
