// Landmark debug page: draws the assembled Kaggle-layout frame over the video.
import { FaceLandmarker, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision'
import { stage1 } from './features'
import { lookupSign } from './lookup'
import { LOW_CONFIDENCE, SignModel } from './model'
import { CameraPicker } from './ui/camera'
import {
  COORDS_PER_POINT as C,
  LandmarkExtractor,
  LandmarkRecorder,
  OFFSETS,
  toRecording,
  toTrainingAspect,
  TRAIN_H_OVER_W,
  type KaggleFrame,
} from './landmarks'
import { featureSpec } from './spec'

type Connection = { start: number; end: number } // not exported by tasks-vision 1.2.3

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const video = $<HTMLVideoElement>('video')
const canvas = $<HTMLCanvasElement>('overlay')
const ctx = canvas.getContext('2d')!
const status = $('status')
const downloadBtn = $<HTMLButtonElement>('download')

const USED = new Set(featureSpec.groups.flatMap((g) => g.global_index))
const UPPER_POSE = new Set([0, 11, 12, 13, 14, 15, 16])
const POSE_EDGES = PoseLandmarker.POSE_CONNECTIONS.filter((c) => UPPER_POSE.has(c.start) && UPPER_POSE.has(c.end))
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

let extractor: LandmarkExtractor
let recorder: LandmarkRecorder | null = null
let sourceName = 'webcam'
let frameTimes: number[] = []

function pt(f: KaggleFrame, g: number): [number, number] | null {
  const x = f[g * C]
  const y = f[g * C + 1]
  return Number.isNaN(x) ? null : [x * canvas.width, y * canvas.height]
}

function edges(f: KaggleFrame, offset: number, conns: Connection[], color: string, width: number) {
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.beginPath()
  for (const { start, end } of conns) {
    const a = pt(f, offset + start)
    const b = pt(f, offset + end)
    if (a && b) {
      ctx.moveTo(a[0], a[1])
      ctx.lineTo(b[0], b[1])
    }
  }
  ctx.stroke()
}

function dots(
  f: KaggleFrame,
  from: number,
  count: number,
  color: string,
  only?: (local: number) => boolean,
  enlargeUsed = true, // false for the face: every face dot gets the small mesh radius
) {
  const r = Math.max(2, canvas.width / 320)
  ctx.fillStyle = color
  for (let i = 0; i < count; i++) {
    if (only && !only(i)) continue
    const g = from + i
    const p = pt(f, g)
    if (!p) continue
    const size = enlargeUsed && USED.has(g) ? r * 1.6 : r * 0.6
    ctx.beginPath()
    ctx.arc(p[0], p[1], size, 0, Math.PI * 2)
    ctx.fill()
  }
}

function label(f: KaggleFrame, g: number, text: string, color: string) {
  const p = pt(f, g)
  if (!p) return
  ctx.font = `${Math.max(12, canvas.width / 50)}px system-ui`
  ctx.fillStyle = color
  // keep text readable when the preview is mirrored with CSS
  const mirrored = $('stage').classList.contains('mirror')
  ctx.save()
  ctx.translate(p[0], p[1] + 24)
  if (mirrored) ctx.scale(-1, 1)
  ctx.fillText(text, 0, 0)
  ctx.restore()
}

/**
 * Canvas pixels = the video's intrinsic size (landmarks are normalized to it), and the canvas CSS box =
 * the rect where the video picture is actually displayed (object-fit: contain inside the element box).
 */
function layoutOverlay(): void {
  const vw = video.videoWidth
  const vh = video.videoHeight
  if (!vw || !vh) return
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw
    canvas.height = vh
  }
  const bw = video.clientWidth
  const bh = video.clientHeight
  const s = Math.min(bw / vw, bh / vh)
  const w = vw * s
  const h = vh * s
  canvas.style.left = `${video.offsetLeft + (bw - w) / 2}px`
  canvas.style.top = `${video.offsetTop + (bh - h) / 2}px`
  canvas.style.width = `${w}px`
  canvas.style.height = `${h}px`
}
new ResizeObserver(layoutOverlay).observe(video)
video.addEventListener('loadedmetadata', layoutOverlay)
video.addEventListener('resize', layoutOverlay) // intrinsic size change (e.g. webcam renegotiation)

/** Draws RAW landmarks (as assembled, no aspect correction): they are normalized to the video frame. */
function draw(f: KaggleFrame) {
  layoutOverlay()
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  const left = css('--left')
  const right = css('--right')
  const face = css('--face')
  const pose = css('--pose')
  const w = Math.max(2, canvas.width / 400)

  // Face (lips, eyes, eyebrows, forehead, chin): one soft neutral color at 50% so the hands stand out.
  // By default only a thin lip outline; individual face points behind the "Show face points" toggle.
  ctx.globalAlpha = 0.5
  edges(f, OFFSETS.face, FaceLandmarker.FACE_LANDMARKS_LIPS, face, Math.max(1, w / 2))
  if ($<HTMLInputElement>('face-points').checked) {
    // All face dots share the small mesh radius; the 60 the model uses are only slightly less faint.
    ctx.globalAlpha = 0.25
    dots(f, OFFSETS.face, 468, face, (i) => !USED.has(OFFSETS.face + i), false)
    ctx.globalAlpha = 0.45
    dots(f, OFFSETS.face, 468, face, (i) => USED.has(OFFSETS.face + i), false)
  }
  ctx.globalAlpha = 1
  edges(f, OFFSETS.pose, POSE_EDGES, pose, w)
  dots(f, OFFSETS.pose, 33, pose, (i) => UPPER_POSE.has(i))
  edges(f, OFFSETS.left_hand, HandLandmarker.HAND_CONNECTIONS, left, w)
  dots(f, OFFSETS.left_hand, 21, left)
  edges(f, OFFSETS.right_hand, HandLandmarker.HAND_CONNECTIONS, right, w)
  dots(f, OFFSETS.right_hand, 21, right)
  label(f, OFFSETS.left_hand, 'left slot', left)
  label(f, OFFSETS.right_hand, 'right slot', right)
}

const present = (f: KaggleFrame, offset: number, n: number) => {
  for (let i = 0; i < n; i++) if (!Number.isNaN(f[(offset + i) * C])) return true
  return false
}
const yesNo = (b: boolean) => (b ? 'detected' : 'missing')

function onFrame(f: KaggleFrame) {
  draw(f)
  const now = performance.now()
  frameTimes.push(now)
  frameTimes = frameTimes.filter((t) => t > now - 2000)
  $('fps').textContent = (frameTimes.length / 2).toFixed(1)
  $('p-left').textContent = yesNo(present(f, OFFSETS.left_hand, 21))
  $('p-right').textContent = yesNo(present(f, OFFSETS.right_hand, 21))
  $('p-lips').textContent = yesNo(present(f, 0, 468))
  $('p-pose').textContent = yesNo(present(f, OFFSETS.pose, 33))
  const W = video.videoWidth
  const H = video.videoHeight
  $('size').textContent = `${W} × ${H} (H/W ${(H / W).toFixed(2)})`
  $('aspect').textContent = `y × ${(H / W / TRAIN_H_OVER_W).toFixed(3)} (training H/W ${TRAIN_H_OVER_W})`
  if (recorder) {
    const recent = recorder.frames().slice(-recorder.fps * 2)
    const s1 = stage1(toTrainingAspect(recent.map((r) => r.frame), W, H))
    $('seq').textContent = `${recent.length} frames → ${s1.T} after trim, ${s1.mirrored ? 'left-dominant (mirrored)' : 'right-dominant'}`
  }
}

function startRecorder() {
  recorder?.stop()
  recorder = new LandmarkRecorder(extractor, video, { fps: 20, bufferSeconds: 6, onFrame: (tf) => onFrame(tf.frame) })
  recorder.start()
  downloadBtn.disabled = false
  status.textContent = `Running on ${sourceName}`
  updateLookupButton()
}

// ---------------------------------------------------------------- sign lookup

const lookupBtn = $<HTMLButtonElement>('lookup')
const lookupStatus = $('lookup-status')
const matchesEl = $('matches')
let model: SignModel | null = null
let busy = false
const RECORD_MS = 2000

function updateLookupButton() {
  lookupBtn.disabled = !model || !recorder || busy
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

async function lookup() {
  if (!model || !recorder) return
  busy = true
  updateLookupButton()
  matchesEl.innerHTML = ''
  try {
    recorder.clear()
    const start = performance.now()
    while (performance.now() - start < RECORD_MS) {
      lookupStatus.textContent = `Recording… ${((RECORD_MS - (performance.now() - start)) / 1000).toFixed(1)} s`
      await sleep(100)
    }
    const frames = recorder.frames().map((f) => f.frame)
    if (frames.length === 0) throw new Error('no frames captured; is the video playing?')
    // Same pipeline as training: aspect correction, then features at the frame length this model expects.
    const corrected = toTrainingAspect(frames, video.videoWidth, video.videoHeight)
    const variants = Number($<HTMLSelectElement>('tta').value)
    const { matches, stage1: s1, timing, variants: used } = await lookupSign(model, corrected, { variants, k: 5 })
    for (const m of matches) {
      const li = document.createElement('li')
      const pct = (m.confidence * 100).toFixed(1)
      li.innerHTML = `<div class="match"><span class="gloss"></span><div class="bar" role="img"><span></span></div><span class="pct">${pct}%</span></div>`
      li.querySelector('.gloss')!.textContent = m.gloss
      const bar = li.querySelector<HTMLElement>('.bar')!
      bar.setAttribute('aria-label', `${pct} percent confidence`)
      bar.querySelector('span')!.style.width = `${Math.max(1, m.confidence * 100)}%`
      matchesEl.appendChild(li)
    }
    const handFrames = s1.T
    const unsure = matches[0].confidence < LOW_CONFIDENCE
    lookupStatus.textContent =
      (unsure ? 'Not sure. Try again with just the one sign inside the 2 seconds. ' : '') +
      `${frames.length} frames recorded, ${handFrames} with hands after trimming` +
      `${s1.mirrored ? ' (left-dominant, mirrored)' : ''}. ` +
      `Lookup ${timing.totalMs.toFixed(0)} ms (features ${timing.featuresMs.toFixed(0)} + model ` +
      `${timing.modelMs.toFixed(0)}), ${used === 1 ? 'single pass' : `averaged over ${used} trimmed versions`}.`
  } catch (e) {
    lookupStatus.textContent = `Lookup failed: ${(e as Error).message}`
  } finally {
    busy = false
    updateLookupButton()
  }
}

lookupBtn.addEventListener('click', () => void lookup())

SignModel.load()
  .then((m) => {
    model = m
    $('model-info').textContent = `Model loaded: ${m.labels.length} signs, expects ${m.length} frames × 222 features (features.ts resamples to ${m.length}).`
    updateLookupButton()
  })
  .catch((e) => {
    $('model-info').textContent = `No model loaded: ${(e as Error).message}. Put model.onnx and labels.json in model/ and restart npm run dev.`
  })

const camera = new CameraPicker($<HTMLSelectElement>('camera'))
void camera.refresh()

async function showStream(stream: MediaStream, label: string) {
  video.srcObject = stream
  video.controls = false
  await video.play()
  sourceName = label
  startRecorder() // same <video> element, so the recorder just keeps going on the new camera
  if (camera.settingsText) status.textContent = `Running on ${label}: ${camera.settingsText}`
}

camera.onSwitch = (stream, label) => void showStream(stream, label)
camera.onError = (message) => (status.textContent = message)

$('webcam').addEventListener('click', async () => {
  try {
    const stream = await camera.open()
    await showStream(stream, camera.label)
  } catch (e) {
    status.textContent = `Webcam unavailable: ${(e as Error).message}`
  }
})

$<HTMLInputElement>('file').addEventListener('change', async (ev) => {
  const file = (ev.target as HTMLInputElement).files?.[0]
  if (!file) return
  camera.close()
  video.srcObject = null
  video.src = URL.createObjectURL(file)
  video.controls = true
  await video.play()
  sourceName = file.name
  startRecorder()
})

$<HTMLInputElement>('mirror').addEventListener('change', (ev) => {
  $('stage').classList.toggle('mirror', (ev.target as HTMLInputElement).checked)
})

downloadBtn.addEventListener('click', () => {
  if (!recorder) return
  const rec = toRecording(recorder.frames(), video.videoWidth, video.videoHeight, recorder.fps, sourceName)
  const url = URL.createObjectURL(new Blob([JSON.stringify(rec)], { type: 'application/json' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `landmarks-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  a.click()
  URL.revokeObjectURL(url)
  status.textContent = `Saved ${rec.frames.length} frames. Put the file in demo/landmarks/ to include it in the parity test.`
})

LandmarkExtractor.create()
  .then((ex) => {
    extractor = ex
    $('delegate').textContent = ex.delegate
    status.textContent = 'Ready. Choose the webcam or a video file.'
  })
  .catch((e) => {
    status.textContent = `MediaPipe failed to load: ${(e as Error).message}. Did scripts/fetch-mediapipe.mjs run?`
  })
