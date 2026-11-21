// Video frames -> landmarks in the Kaggle 543-point layout, via @mediapipe/tasks-vision.
// Shared with the extension (imported as @shared/landmarks).
//
// Kaggle layout per frame: face 0-467 | left_hand 468-488 | pose 489-521 | right_hand 522-542.
// FaceLandmarker returns 478 points; the last 10 are iris points and are dropped.
// Hands go into slots by nearest pose wrist (feature_spec.hand_assignment), not by handedness label.
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import {
  COORDS_PER_POINT,
  emptyFrame,
  frameToNested,
  OFFSETS,
  SIZES,
  type KaggleFrame,
  type LandmarkRecording,
} from './layout'
import { featureSpec } from './spec'

export * from './layout'

const C = COORDS_PER_POINT
const LEFT_WRIST = featureSpec.hand_assignment.left_hand_slot_wrist.landmark_index // pose 15
const RIGHT_WRIST = featureSpec.hand_assignment.right_hand_slot_wrist.landmark_index // pose 16
const HAND_POINT = featureSpec.hand_assignment.hand_point.landmark_index // hand 0 (wrist)
export const TRAIN_H_OVER_W: number = featureSpec.input_aspect.train_H_over_W

// ---------------------------------------------------------------- assembly (pure, testable)

export interface RawDetections {
  hands: NormalizedLandmark[][]
  /** 'Left' | 'Right' per hand, the subject's hand for an unmirrored image (tasks-vision 1.2.3 docs). */
  handedness: string[]
  pose?: NormalizedLandmark[]
  face?: NormalizedLandmark[]
}

function put(frame: KaggleFrame, offset: number, pts: NormalizedLandmark[], count: number): void {
  for (let i = 0; i < Math.min(count, pts.length); i++) {
    const o = (offset + i) * C
    frame[o] = pts[i].x
    frame[o + 1] = pts[i].y
    frame[o + 2] = pts[i].z
  }
}

const dist = (a: NormalizedLandmark, b: NormalizedLandmark) => Math.hypot(a.x - b.x, a.y - b.y)

/** Returns [slot of hand 0, slot of hand 1], 'L' = left_hand slot, 'R' = right_hand slot. */
export function assignHands(d: RawDetections): ('L' | 'R')[] {
  const n = Math.min(2, d.hands.length)
  const p = d.pose
  if (n === 0) return []
  if (p && p.length > Math.max(LEFT_WRIST, RIGHT_WRIST)) {
    const wl = p[LEFT_WRIST]
    const wr = p[RIGHT_WRIST]
    const w0 = d.hands[0][HAND_POINT]
    if (n === 1) return [dist(w0, wl) <= dist(w0, wr) ? 'L' : 'R']
    const w1 = d.hands[1][HAND_POINT]
    return dist(w0, wl) + dist(w1, wr) <= dist(w0, wr) + dist(w1, wl) ? ['L', 'R'] : ['R', 'L']
  }
  // No pose: fall back to the handedness label; resolve a clash by image side
  // (the subject's left hand appears on the image's right in unmirrored video).
  const labels = d.handedness.slice(0, n).map((h) => (h === 'Left' ? 'L' : 'R') as 'L' | 'R')
  if (n === 2 && labels[0] === labels[1]) {
    return d.hands[0][HAND_POINT].x >= d.hands[1][HAND_POINT].x ? ['L', 'R'] : ['R', 'L']
  }
  return labels
}

export function assembleFrame(d: RawDetections): KaggleFrame {
  const frame = emptyFrame()
  if (d.face) put(frame, OFFSETS.face, d.face, SIZES.face) // first 468 of 478: drops the iris points
  if (d.pose) put(frame, OFFSETS.pose, d.pose, SIZES.pose)
  assignHands(d).forEach((slot, i) => {
    put(frame, slot === 'L' ? OFFSETS.left_hand : OFFSETS.right_hand, d.hands[i], SIZES.left_hand)
  })
  return frame
}

/**
 * Rescale y so landmarks from a W x H frame look like the ~3:4 portrait training video
 * (feature_spec.input_aspect). Apply to a copy before computeFeatures; keep raw frames for drawing.
 */
export function toTrainingAspect(frames: KaggleFrame[], width: number, height: number): KaggleFrame[] {
  const k = height / width / TRAIN_H_OVER_W
  return frames.map((f) => {
    const g = f.slice()
    for (let i = 1; i < g.length; i += C) g[i] = 0.5 + (g[i] - 0.5) * k
    return g
  })
}

// ---------------------------------------------------------------- MediaPipe

export interface ExtractorOptions {
  /** Folder with wasm/ and the .task models (see scripts/fetch-mediapipe.mjs). */
  assetsBase?: string
  delegate?: 'GPU' | 'CPU'
  poseModel?: 'pose_landmarker_full.task' | 'pose_landmarker_lite.task'
}

export class LandmarkExtractor {
  private lastTs = 0

  private constructor(
    private readonly hand: HandLandmarker,
    private readonly pose: PoseLandmarker,
    private readonly face: FaceLandmarker,
    readonly delegate: 'GPU' | 'CPU',
  ) {}

  static async create(opts: ExtractorOptions = {}): Promise<LandmarkExtractor> {
    const base = opts.assetsBase ?? `${import.meta.env.BASE_URL}mediapipe`
    const fileset = await FilesetResolver.forVisionTasks(`${base}/wasm`)
    const build = (delegate: 'GPU' | 'CPU') =>
      Promise.all([
        HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${base}/hand_landmarker.task`, delegate },
          runningMode: 'VIDEO',
          numHands: 2,
        }),
        PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${base}/${opts.poseModel ?? 'pose_landmarker_full.task'}`, delegate },
          runningMode: 'VIDEO',
          numPoses: 1,
        }),
        FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${base}/face_landmarker.task`, delegate },
          runningMode: 'VIDEO',
          numFaces: 1,
        }),
      ])
    const first = opts.delegate ?? 'GPU'
    try {
      const [h, p, f] = await build(first)
      return new LandmarkExtractor(h, p, f, first)
    } catch (e) {
      if (first === 'CPU') throw e
      console.warn('MediaPipe GPU delegate failed, using CPU', e)
      const [h, p, f] = await build('CPU')
      return new LandmarkExtractor(h, p, f, 'CPU')
    }
  }

  /** Last timestamp passed to MediaPipe (ms). VIDEO mode needs strictly increasing timestamps. */
  get lastTimestamp(): number {
    return this.lastTs
  }

  detectRaw(source: TexImageSource, timestampMs: number = performance.now()): RawDetections {
    const ts = Math.max(this.lastTs + 1, Math.round(timestampMs))
    this.lastTs = ts
    const h = this.hand.detectForVideo(source, ts)
    const p = this.pose.detectForVideo(source, ts)
    const f = this.face.detectForVideo(source, ts)
    return {
      hands: h.landmarks,
      handedness: h.handedness.map((c) => c[0]?.categoryName ?? ''),
      pose: p.landmarks[0],
      face: f.faceLandmarks[0],
    }
  }

  detect(source: TexImageSource, timestampMs?: number): KaggleFrame {
    return assembleFrame(this.detectRaw(source, timestampMs))
  }

  close(): void {
    this.hand.close()
    this.pose.close()
    this.face.close()
  }
}

// ---------------------------------------------------------------- capture

export interface TimedFrame {
  /** Media time, seconds. */
  t: number
  frame: KaggleFrame
}

export interface RecorderOptions {
  fps?: number
  bufferSeconds?: number
  onFrame?: (f: TimedFrame, raw: RawDetections) => void
}

/** Runs the extractor on a playing <video> (file or webcam) at ~fps, keeping a rolling buffer. */
export class LandmarkRecorder {
  readonly fps: number
  private readonly bufferSeconds: number
  private buf: (TimedFrame & { wall: number })[] = []
  private running = false
  private due = 0

  constructor(
    private readonly extractor: LandmarkExtractor,
    private readonly video: HTMLVideoElement,
    private readonly opts: RecorderOptions = {},
  ) {
    this.fps = opts.fps ?? 20
    this.bufferSeconds = opts.bufferSeconds ?? 6
  }

  start(): void {
    if (this.running) return
    this.running = true
    this.due = 0
    this.schedule()
  }

  stop(): void {
    this.running = false
  }

  clear(): void {
    this.buf = []
  }

  frames(): TimedFrame[] {
    return this.buf.map(({ t, frame }) => ({ t, frame }))
  }

  private schedule(): void {
    if (!this.running) return
    const v = this.video
    if ('requestVideoFrameCallback' in v) {
      v.requestVideoFrameCallback((now, meta) => this.tick(now, meta.mediaTime))
    } else {
      requestAnimationFrame((now) => this.tick(now, (v as HTMLVideoElement).currentTime))
    }
  }

  private tick(now: number, mediaTime: number): void {
    if (!this.running) return
    const interval = 1000 / this.fps
    if (this.video.readyState >= 2 && now >= this.due) {
      // keep a steady average rate; don't try to catch up after a stall
      this.due = this.due < now - interval ? now + interval : this.due + interval
      const raw = this.extractor.detectRaw(this.video, now)
      const tf = { t: mediaTime, frame: assembleFrame(raw), wall: now }
      this.buf.push(tf)
      while (this.buf.length && this.buf[0].wall < now - this.bufferSeconds * 1000) this.buf.shift()
      this.opts.onFrame?.({ t: tf.t, frame: tf.frame }, raw)
    }
    this.schedule()
  }
}

function seek(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const done = () => {
      video.removeEventListener('error', fail)
      resolve()
    }
    const fail = () => reject(new Error('video seek failed'))
    video.addEventListener('seeked', done, { once: true })
    video.addEventListener('error', fail, { once: true })
    video.currentTime = t
  })
}

/** Frame-accurate extraction of [startSec, endSec) from a paused, seekable video (uploaded clips). */
export async function extractClip(
  extractor: LandmarkExtractor,
  video: HTMLVideoElement,
  startSec: number,
  endSec: number,
  fps = 20,
  onProgress?: (done: number, total: number) => void,
): Promise<TimedFrame[]> {
  video.pause()
  const n = Math.max(1, Math.round((endSec - startSec) * fps))
  const base = extractor.lastTimestamp + 1000 // gap so trackers don't smooth across clips
  const out: TimedFrame[] = []
  for (let i = 0; i < n; i++) {
    const t = Math.min(startSec + i / fps, Math.max(0, video.duration - 1e-3))
    await seek(video, t)
    out.push({ t, frame: extractor.detect(video, base + (i * 1000) / fps) })
    onProgress?.(i + 1, n)
  }
  return out
}

export function toRecording(frames: TimedFrame[], width: number, height: number, fps: number, source: string): LandmarkRecording {
  return {
    format: 'signseek-landmarks-v1',
    source,
    width,
    height,
    fps,
    t: frames.map((f) => Math.round(f.t * 1000) / 1000),
    frames: frames.map((f) => frameToNested(f.frame)),
  }
}
