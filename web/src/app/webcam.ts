// Webcam screen: camera picker, live preview with a "hands detected" badge, and Record:
// 3-2-1 countdown, 2 s of landmarks (and a video recording for the replay), then the lookup.
import { COORDS_PER_POINT as C, LandmarkRecorder, OFFSETS, toTrainingAspect, type KaggleFrame } from '../landmarks'
import { lookupSign } from '../lookup'
import type { SignModel } from '../model'
import { CameraPicker } from '../ui/camera'
import { getExtractor, getModel, loadReferenceData } from './engine'
import { renderResults } from './results'
import { onEnter, onLeave, show } from './router'

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

const picker = new CameraPicker($<HTMLSelectElement>('camera'))
void picker.refresh()
let stream: MediaStream | null = null
let recorder: LandmarkRecorder | null = null
let model: SignModel | null = null
let busy = false
let session = 0 // bumped when leaving the screen: stale async work checks it and stops
let recordWhenReady = false
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
    recorder = new LandmarkRecorder(extractor, video, { fps: 20, bufferSeconds: 6, onFrame: (tf) => showHands(hasHands(tf.frame)) })
    recorder.start()
    status.textContent = ''
    updateButton()
    if (recordWhenReady) {
      recordWhenReady = false
      void record()
    }
  } catch (e) {
    if (my === session) status.textContent = `Could not load the sign model or hand tracking: ${(e as Error).message}`
  }
}

function leave() {
  session++
  recordWhenReady = false
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
    const frames = recorder.frames().map((f) => f.frame)
    if (frames.length === 0) throw new Error('no video frames were captured')

    busyEl.hidden = false
    status.textContent = 'Finding matches…'
    const corrected = toTrainingAspect(frames, video.videoWidth, video.videoHeight)
    const res = await lookupSign(model, corrected, { variants: TTA_VARIANTS, k: 5 })
    if (my !== session) return
    if (res.handFrames === 0) {
      status.textContent = 'No hands were seen in the recording. Make sure your hands are in view, then try again.'
      return
    }
    if (lastClipUrl) URL.revokeObjectURL(lastClipUrl)
    lastClipUrl = blob ? URL.createObjectURL(blob) : null
    renderResults({ matches: res.matches, source: 'webcam', clipUrl: lastClipUrl, seconds: RECORD_MS / 1000, mirror: true })
    show('results')
  } catch (e) {
    if (my === session) status.textContent = `Lookup failed: ${(e as Error).message}`
  } finally {
    recordBtn.classList.remove('recording')
    recordLabel.textContent = 'Record 2 seconds'
    overlay(null)
    busyEl.hidden = true
    if (my === session) {
      busy = false
      updateButton()
    }
  }
}

/** From the results screen: go back to the webcam and record again as soon as it is ready. */
export function recordAgain() {
  recordWhenReady = true
  show('webcam')
}

recordBtn.addEventListener('click', () => void record())
onEnter('webcam', () => void enter())
onLeave('webcam', leave)
