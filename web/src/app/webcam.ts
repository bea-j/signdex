// Webcam screen: camera picker, live preview with a "hands detected" badge, and Record:
// 3-2-1 countdown, 2 s of landmarks (and a video recording for the replay), then the lookup.
import { COORDS_PER_POINT as C, LandmarkRecorder, OFFSETS, toTrainingAspect, type KaggleFrame, type TimedFrame } from '../landmarks'
import { lookupSign, type LookupResult } from '../lookup'
import type { SignMatch, SignModel } from '../model'
import { CameraPicker } from '../ui/camera'
import { frameAt, LandmarkOverlay } from '../ui/overlay'
import { prefs } from '../ui/prefs'
import { frameStats, lookupStats, partStats, renderDetails } from './details'
import { getExtractor, getModel, loadReferenceData } from './engine'
import { renderCards } from './results'
import { bgBusy } from './background'
import { onEnter, onLeave } from './router'

export const RECORD_MS = 2000
const COUNTDOWN_FROM = 3
const TTA_VARIANTS = 5

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

function updateButton() {
  recordBtn.disabled = busy || !recorder || !model
}

async function attach(s: MediaStream) {
  stream = s
  video.srcObject = s
  await video.play()
  empty.hidden = true
}

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
  status.textContent = 'Loading hand tracking and the sign model…'
  updateButton()
  const engine = Promise.all([getExtractor(), getModel(), loadReferenceData()])
  engine.catch(() => {}) // handled below; avoid an unhandled rejection while the camera opens

  try {
    const s = await picker.open()
    if (my !== session) return picker.close()
    await attach(s)
  } catch (e) {
    if (my !== session) return
    empty.textContent = `Camera unavailable: ${(e as Error).message}. Allow camera access, or upload a clip instead.`
    status.textContent = ''
    return
  }

  try {
    const [extractor, m] = await engine
    if (my !== session) return
    model = m
    delegate = extractor.delegate
    recorder = new LandmarkRecorder(extractor, video, { fps: 20, bufferSeconds: 6, onFrame: (tf) => {
        showHands(hasHands(tf.frame))
        liveOverlay.draw(tf.frame, layers())
        const now = performance.now()
        frameTimes.push(now)
        while (frameTimes.length && frameTimes[0] < now - 2000) frameTimes.shift()
        lastFrame = tf.frame
        scheduleDetails()
      },
    })
    recorder.start()
    status.textContent = ''
    updateButton()
  } catch (e) {
    if (my === session) status.textContent = `Could not load the sign model or hand tracking: ${(e as Error).message}`
  }
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
      return stopped
    }
  } catch {
    return null
  }
}

export async function record() {
  if (busy || !recorder || !model) return
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
    const blob = (await stopClip?.()) ?? null
    await finishLookup(recorder.frames(), blob, video.videoWidth, video.videoHeight, my)
  } catch (e) {
    if (my === session) status.textContent = `Lookup failed: ${(e as Error).message}`
  } finally {
    recordBtn.classList.remove('recording')
    recordLabel.textContent = hasResult ? 'Record again' : 'Record 2 seconds'
    overlay(null)
    busyEl.hidden = true
    if (my === session) {
      busy = false
      updateButton()
    }
  }
}

/** Recorded landmarks (+ the clip for the replay) -> lookup -> results panel. */
async function finishLookup(timed: TimedFrame[], blob: Blob | null, width: number, height: number, my: number) {
  const frames = timed.map((f) => f.frame)
  if (frames.length === 0) throw new Error('no video frames were captured')
  replayFrames = timed.map((f) => ({ t: f.t - timed[0].t, frame: f.frame }))
  busyEl.hidden = false
  status.textContent = 'Finding matches…'
  const corrected = toTrainingAspect(frames, width, height)
  const res = await lookupSign(model!, corrected, { variants: TTA_VARIANTS, k: 5 })
  if (my !== session) return
  if (res.handFrames === 0) {
    status.textContent = 'No hands were seen in the recording. Make sure your hands are in view, then try again.'
    return
  }
  if (lastClipUrl) URL.revokeObjectURL(lastClipUrl)
  lastClipUrl = blob ? URL.createObjectURL(blob) : null
  lastLookup = { recorded: frames.length, res }
  showResults(res.matches, lastClipUrl)
  scheduleDetails()
  status.textContent = ''
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
