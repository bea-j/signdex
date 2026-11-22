// Webcam screen: camera picker, live preview with a "hands detected" badge, and Record:
// 3-2-1 countdown, 2 s of landmarks (and a video recording for the replay), then the lookup.
// Record shows the loading progress until the model and hand tracking are ready, and offers a retry if loading
// fails. Nothing here waits forever: video.play(), the clip recorder and the downloads all have time limits.
import { COORDS_PER_POINT as C, LandmarkRecorder, OFFSETS, toTrainingAspect, type KaggleFrame, type TimedFrame } from '../landmarks'
import { lookupSign, type LookupResult } from '../lookup'
import type { SignMatch, SignModel } from '../model'
import { CameraPicker } from '../ui/camera'
import { frameAt, LandmarkOverlay } from '../ui/overlay'
import { prefs } from '../ui/prefs'
import { frameStats, lookupStats, partStats, renderDetails } from './details'
import { diag } from './diag'
import { loadEngine, loadState, onLoadState, useCpuExtractor, type LoadState } from './engine'
import type { LandmarkExtractor } from '../landmarks'
import { renderCards } from './results'
import { bgBusy } from './background'
import { onEnter, onLeave } from './router'

export const RECORD_MS = 2000
const COUNTDOWN_FROM = 3
const TTA_VARIANTS = 5
/** Fewer tracked frames than this in a recording: the results get a reliability note. */
const MIN_GOOD_FRAMES = 12

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const video = $<HTMLVideoElement>('webcam-video')
const recordBtn = $<HTMLButtonElement>('record')
const recordLabel = $('record-label')
const status = $('webcam-status')
const empty = $('webcam-empty')
const badge = $('hands-badge')
const handsText = $('hands-text')
const countdownEl = $('webcam-countdown')
const busyEl = $('webcam-busy')
const stage = $('webcam-stage')

const replay = $<HTMLVideoElement>('webcam-replay')
const liveOverlay = new LandmarkOverlay(video)
const replayOverlay = new LandmarkOverlay(replay)
const layers = () => ({ points: prefs.get('points'), face: prefs.get('face') })
/** Landmarks of the last recording, t = seconds from the start of the recorded clip. */
let replayFrames: { t: number; frame: KaggleFrame }[] = []

const picker = new CameraPicker($<HTMLSelectElement>('camera'))
void picker.refresh()
let stream: MediaStream | null = null
let recorder: LandmarkRecorder | null = null
let model: SignModel | null = null
let busy = false
let session = 0 // bumped when leaving the screen: stale async work checks it and stops
let hasResult = false
// details panel
let delegate = ''
let frameTimes: number[] = []
let lastFrame: KaggleFrame | null = null
let lastLookup: { recorded: number; res: LookupResult } | null = null
let lastClipUrl: string | null = null

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

function overlay(text: string | null, rec = false) {
  countdownEl.hidden = text === null
  countdownEl.classList.toggle('rec', rec)
  countdownEl.replaceChildren()
  if (text !== null) countdownEl.append(Object.assign(document.createElement('span'), { textContent: text }))
}

function hasHands(f: KaggleFrame): boolean {
  for (const off of [OFFSETS.left_hand, OFFSETS.right_hand])
    for (let i = 0; i < 21; i++) if (!Number.isNaN(f[(off + i) * C])) return true
  return false
}

let handsShown: boolean | null = null
function showHands(on: boolean) {
  if (on === handsShown) return
  handsShown = on
  badge.hidden = false
  badge.classList.toggle('warn', !on)
  handsText.textContent = on ? 'Hands detected' : 'Looking for hands'
}

const restLabel = () => (hasResult ? 'Record again' : 'Record 2 seconds')

/** Record's state: loading (with progress), failed (tap to retry), waiting for the camera, ready or busy. */
function updateButton() {
  if (busy) {
    recordBtn.disabled = true
    return
  }
  const load: LoadState = loadState()
  const failed = !model && load.phase === 'failed'
  const loading = !model && !failed
  recordBtn.disabled = loading || (!!model && !recorder)
  recordBtn.classList.toggle('loading', loading)
  recordBtn.setAttribute('aria-busy', String(loading))
  recordBtn.style.setProperty('--load', loading ? String(load.fraction) : '0')
  recordLabel.textContent = failed ? 'Try loading again' : loading ? `Loading… ${Math.round(load.fraction * 100)}%` : restLabel()
}

onLoadState((s) => {
  if ($('screen-webcam').hidden || model) return
  updateButton()
  if (s.phase === 'loading') status.textContent = `Loading hand tracking and the sign model (first visit: about 40 MB)… ${s.step}`
})

async function attach(s: MediaStream) {
  stream = s
  video.srcObject = s
  // iOS: muted + playsinline + autoplay let the preview play inline. Don't let play() hold anything up.
  try {
    await Promise.race([video.play(), sleep(4000)])
  } catch (e) {
    diag.error(`video.play(): ${(e as Error).message}`)
  }
  empty.hidden = !video.paused
  if (video.paused) empty.textContent = 'Tap here to start the camera preview.'
  showCameraInfo()
  // phones: bring the camera into view (Record stays on screen, see app.css)
  if (matchMedia('(pointer: coarse)').matches && scrollY < 40 && stage.getBoundingClientRect().bottom > innerHeight)
    stage.scrollIntoView({ block: 'start', behavior: 'smooth' })
}
stage.addEventListener('click', () => {
  if (video.srcObject && video.paused) void video.play().then(() => (empty.hidden = true), () => {})
})

/** The stage takes the camera's real shape (a phone may deliver portrait video, e.g. 720 x 1280). */
function showCameraInfo() {
  const w = video.videoWidth
  const h = video.videoHeight
  if (w && h) stage.style.setProperty('--video-aspect', `${w} / ${h}`)
  diag.set('Camera', 'Track', picker.settingsText)
  diag.set('Camera', 'Video element', w ? `${w} × ${h} (${w > h ? 'landscape' : w < h ? 'portrait' : 'square'})` : 'no frames yet')
  const box = video.getBoundingClientRect()
  diag.set('Camera', 'Shown at', `${Math.round(box.width)} × ${Math.round(box.height)} CSS px, object-fit contain`)
  diag.set('Camera', 'Playing', video.paused ? 'paused' : 'yes')
}
video.addEventListener('loadedmetadata', showCameraInfo)
video.addEventListener('resize', showCameraInfo)

picker.onSwitch = (s) => void attach(s) // same <video>, so the landmark recorder keeps running
picker.onError = (m) => (status.textContent = m)

async function enter() {
  const my = ++session
  busy = false
  handsShown = null
  badge.hidden = true
  overlay(null)
  busyEl.hidden = true
  empty.hidden = false
  empty.textContent = 'Starting the camera…'
  status.textContent = model ? '' : 'Loading hand tracking and the sign model…'
  updateButton()
  const engine = loadEngine()
  engine.catch(() => {}) // handled below; avoid an unhandled rejection while the camera opens

  try {
    const s = await picker.open()
    if (my !== session) return picker.close()
    await attach(s)
  } catch (e) {
    if (my !== session) return
    diag.error(`camera: ${(e as Error).name}: ${(e as Error).message}`)
    empty.textContent = `Camera unavailable: ${(e as Error).message}. Allow camera access, or upload a clip instead.`
    status.textContent = ''
    return
  }
  await startTracking(my)
}

/** Wait for the model and trackers, then track the live video. Also the retry after a failed load. */
async function startTracking(my: number) {
  updateButton()
  try {
    const [extractor, m] = await loadEngine()
    if (my !== session) return
    model = m
    startRecorder(extractor)
    status.textContent = ''
  } catch (e) {
    if (my === session)
      status.textContent = `Could not load the sign model or hand tracking: ${(e as Error).message}. Tap “Try loading again”, or reload the page.`
  } finally {
    if (my === session) updateButton()
  }
}

let detectErrors = 0
function startRecorder(extractor: LandmarkExtractor) {
  recorder?.stop()
  delegate = extractor.delegate
  detectErrors = 0
  const my = session
  recorder = new LandmarkRecorder(extractor, video, {
    fps: 20,
    bufferSeconds: 6,
    onFrame: (tf) => {
      detectErrors = 0
      showHands(hasHands(tf.frame))
      liveOverlay.draw(tf.frame, layers())
      const now = performance.now()
      frameTimes.push(now)
      while (frameTimes.length && frameTimes[0] < now - 2000) frameTimes.shift()
      lastFrame = tf.frame
      scheduleDetails()
      if (diag.enabled) {
        diag.set('Hand tracking', 'Landmark rate', `${(frameTimes.length / 2).toFixed(1)} fps, last detection ${Math.round(recorder?.lastDetectMs ?? 0)} ms`)
        diag.set('Hand tracking', 'Frame size', `${tf.width} × ${tf.height}`)
      }
    },
    onError: (e) => {
      detectErrors++
      diag.error(`detection (${extractor.delegate}): ${(e as Error)?.message ?? e}`)
      // the GPU keeps failing while tracking (it can set up fine and fail later, e.g. iOS WebGL): use the CPU
      if (detectErrors === 5) switchToCpu(extractor, my, 'detections failed')
    },
  })
  recorder.start()
  extractor.onGpuLost(() => switchToCpu(extractor, my, 'GPU context lost'))
}

function switchToCpu(extractor: LandmarkExtractor, my: number, why: string) {
  if (extractor.delegate !== 'GPU' || my !== session) return
  if (busy) return void setTimeout(() => switchToCpu(extractor, my, why), 500) // after the lookup in progress
  diag.error(`switching hand tracking to CPU: ${why}`)
  recorder?.stop()
  recorder = null
  updateButton()
  status.textContent = 'Hand tracking had trouble with the graphics chip. Switching to the slower CPU mode…'
  void useCpuExtractor().then(
    (cpu) => {
      if (my !== session) return
      startRecorder(cpu)
      status.textContent = ''
      updateButton()
    },
    (err) => my === session && (status.textContent = `Hand tracking stopped working: ${(err as Error).message}`),
  )
}

function leave() {
  session++
  replay.pause()
  liveOverlay.clear()
  recorder?.stop()
  recorder = null
  picker.close()
  stream = null
  video.srcObject = null
  overlay(null)
  busy = false
}

/** Start a video recording of the stream for the replay on the results screen (best effort). */
function startClipRecording(): (() => Promise<Blob | null>) | null {
  if (!stream || typeof MediaRecorder === 'undefined') return null
  try {
    const mr = new MediaRecorder(stream)
    const chunks: Blob[] = []
    mr.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    const stopped = new Promise<Blob | null>((r) => (mr.onstop = () => r(chunks.length ? new Blob(chunks, { type: mr.mimeType }) : null)))
    mr.start()
    return () => {
      if (mr.state !== 'inactive') mr.stop()
      // the replay is optional: never let a recorder that doesn't stop hold up the lookup
      return Promise.race([stopped, sleep(2000).then(() => null)])
    }
  } catch {
    return null
  }
}

export async function record() {
  if (busy) return
  if (!model && loadState().phase === 'failed') return void startTracking(session) // "Try loading again"
  if (!recorder || !model) return
  if (video.paused) void video.play().catch(() => {}) // a tap is a user gesture: iOS may need it to play
  const my = session
  busy = true
  updateButton()
  try {
    for (let n = COUNTDOWN_FROM; n > 0; n--) {
      overlay(String(n))
      recordLabel.textContent = `Get ready… ${n}`
      status.textContent = `Recording starts in ${n}`
      await sleep(1000)
      if (my !== session) return
    }
    recorder.clear()
    const stopClip = startClipRecording()
    const t0 = performance.now()
    recordBtn.classList.add('recording')
    while (performance.now() - t0 < RECORD_MS) {
      const left = ((RECORD_MS - (performance.now() - t0)) / 1000).toFixed(1)
      overlay(`● REC ${left} s`, true)
      recordLabel.textContent = `Recording… ${left} s`
      status.textContent = 'Recording: sign now'
      await sleep(100)
      if (my !== session) {
        void stopClip?.()
        return
      }
    }
    recordBtn.classList.remove('recording')
    overlay(null)
    const frames = recorder.frames()
    const blob = (await stopClip?.()) ?? null
    await finishLookup(frames, blob, video.videoWidth, video.videoHeight, my)
  } catch (e) {
    if (my === session) status.textContent = `Lookup failed: ${(e as Error).message}`
  } finally {
    recordBtn.classList.remove('recording')
    overlay(null)
    busyEl.hidden = true
    busy = false
    updateButton()
  }
}

/** Recorded landmarks (+ the clip for the replay) -> lookup -> results panel. */
async function finishLookup(timed: TimedFrame[], blob: Blob | null, width: number, height: number, my: number) {
  const frames = timed.map((f) => f.frame)
  if (frames.length === 0) throw new Error('no video frames were captured')
  replayFrames = timed.map((f) => ({ t: f.t - timed[0].t, frame: f.frame }))
  busyEl.hidden = false
  status.textContent = 'Finding matches…'
  // each frame with the size it was captured at (the real track size, portrait or landscape)
  const corrected = timed.map((f) => toTrainingAspect([f.frame], f.width || width, f.height || height)[0])
  diag.set('Last lookup', 'Frames', `${timed.length}, captured at ${[...new Set(timed.map((f) => `${f.width || width}×${f.height || height}`))].join(', ')}`)
  const res = await lookupSign(model!, corrected, { variants: TTA_VARIANTS, k: 5 })
  if (my !== session) return
  diag.set('Last lookup', 'Hands in', `${res.handFrames} of ${frames.length} frames; top: ${res.matches[0]?.label ?? '–'}`)
  if (res.handFrames === 0) {
    status.textContent = 'No hands were seen in the recording. Make sure your hands are in view, then try again.'
    return
  }
  if (lastClipUrl) URL.revokeObjectURL(lastClipUrl)
  lastClipUrl = blob ? URL.createObjectURL(blob) : null
  lastLookup = { recorded: frames.length, res }
  showResults(res.matches, lastClipUrl)
  scheduleDetails()
  // a slow device may track only a few frames in 2 s: say the result is shaky rather than hide it
  status.textContent =
    frames.length < MIN_GOOD_FRAMES
      ? `This device tracked only ${frames.length} frames in the 2 seconds, so these matches are less reliable.`
      : ''
  busyEl.hidden = true
}

/** Results panel next to the camera: replay of the recorded clip and the top 5, updated in place. */
function showResults(matches: SignMatch[], clipUrl: string | null) {
  hasResult = true
  $('webcam-results-empty').hidden = true
  $('webcam-note').hidden = false
  const box = $('webcam-replay-box')
  box.hidden = !clipUrl
  if (clipUrl) {
    replay.addEventListener(
      'loadedmetadata',
      () => replay.videoWidth && box.querySelector<HTMLElement>('.stage')!.style.setProperty('--video-aspect', `${replay.videoWidth} / ${replay.videoHeight}`),
      { once: true },
    )
    replay.src = clipUrl
    void replay.play().catch(() => {})
  }
  $('webcam-caption').textContent = `Your ${(RECORD_MS / 1000).toFixed(1)} s clip, on a loop`
  const cards = $('webcam-cards')
  renderCards(cards, matches, 'Not sure. Try again with just the one sign inside the 2 seconds.')
  // phones: the panel is below the camera, bring it into view
  if (matchMedia('(max-width: 1023px)').matches) $('webcam-results-title').scrollIntoView({ behavior: 'smooth', block: 'start' })
}

recordBtn.addEventListener('click', () => void record())

// ---------------------------------------------------------------- details panel (about 4 updates/s)
const detailsEl = $('webcam-details')
let detailsQueued = false
function scheduleDetails() {
  if (!prefs.get('details') || detailsQueued) return
  detailsQueued = true
  setTimeout(() => {
    detailsQueued = false
    renderDetails(detailsEl, [
      {
        title: 'Camera',
        stats: [
          { label: 'Camera', value: picker.isOpen ? picker.label : 'not open' },
          { label: 'Resolution', value: picker.settingsText || '–' },
          { label: 'Hand tracking', value: delegate ? `MediaPipe on ${delegate}` : 'loading…' },
          { label: 'Landmark rate', value: frameTimes.length ? `${(frameTimes.length / 2).toFixed(1)} fps` : '–' },
          ...frameStats(video.videoWidth, video.videoHeight),
        ],
      },
      { title: 'Live', stats: lastFrame ? partStats([lastFrame], true) : [] },
      { title: 'Last lookup', stats: lastLookup && model ? lookupStats(lastLookup.recorded, lastLookup.res, model.length) : [] },
    ])
  }, 250)
}
prefs.onChange(() => {
  detailsEl.hidden = !prefs.get('details')
  scheduleDetails()
})
detailsEl.hidden = !prefs.get('details')

// replay overlay: the recorded landmarks, in step with the looping clip
function replayTick(_now: number, meta: VideoFrameCallbackMetadata) {
  replayOverlay.draw(frameAt(replayFrames, meta.mediaTime)?.frame ?? null, layers())
  replay.requestVideoFrameCallback(replayTick)
}
const replayEl: HTMLVideoElement = replay
if ('requestVideoFrameCallback' in replayEl) replayEl.requestVideoFrameCallback(replayTick)
else (replayEl as HTMLVideoElement).addEventListener('timeupdate', () =>
  replayOverlay.draw(frameAt(replayFrames, (replayEl as HTMLVideoElement).currentTime)?.frame ?? null, layers()),
)
prefs.onChange(() => {
  if (!prefs.get('points') && !prefs.get('face')) liveOverlay.clear()
  replayOverlay.draw(frameAt(replayFrames, replay.currentTime)?.frame ?? null, layers())
})
// the webcam screen runs MediaPipe the whole time (live tracking, recording, lookup): background paused meanwhile
onEnter('webcam', () => {
  bgBusy(true)
  void enter()
})
onLeave('webcam', () => {
  leave()
  bgBusy(false)
})

// Dev-server-only test hook (removed from production builds): lets the screenshot tests draw real landmark
// sequences, since a headless browser's fake camera has no person in it.
if (import.meta.env.DEV) {
  Object.assign(window, {
    __signdexWebcam: {
      stopLive: () => recorder?.stop(),
      drawLive: (f: KaggleFrame) => liveOverlay.draw(f, layers()),
      /** Run the real lookup path on given landmark frames (as if just recorded), with a clip for the replay. */
      lookup: (timed: TimedFrame[], blob: Blob | null, width: number, height: number) =>
        finishLookup(timed, blob, width, height, session),
      setReplay: (frames: { t: number; frame: KaggleFrame }[], url: string) => {
        replayFrames = frames
        $('webcam-replay-box').hidden = false
        replay.src = url
        void replay.play()
      },
    },
  })
}
