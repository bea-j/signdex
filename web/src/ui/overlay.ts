// Landmark overlay for a <video>: the debug page's drawing (left-hand slot orange, right-hand slot blue, subtle gray
// pose, small faint face points), on a canvas sized to the video's displayed picture (object-fit: contain).
// Put the video and the overlay in the same wrapper; mirror the wrapper (CSS) and both flip together.
import { FaceLandmarker, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision'
import { COORDS_PER_POINT as C, OFFSETS, type KaggleFrame } from '../layout'
import { featureSpec } from '../spec'

type Connection = { start: number; end: number }
export interface OverlayLayers {
  /** Hands and upper-body pose. */
  points: boolean
  /** Lips outline and face points. */
  face: boolean
}

const USED = new Set(featureSpec.groups.flatMap((g) => g.global_index))
const UPPER_POSE = new Set([0, 11, 12, 13, 14, 15, 16])
const POSE_EDGES = PoseLandmarker.POSE_CONNECTIONS.filter((c) => UPPER_POSE.has(c.start) && UPPER_POSE.has(c.end))
const css = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback

export class LandmarkOverlay {
  readonly canvas: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D

  constructor(private readonly video: HTMLVideoElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.className = 'overlay'
    this.canvas.setAttribute('aria-hidden', 'true')
    video.after(this.canvas)
    this.ctx = this.canvas.getContext('2d')!
    new ResizeObserver(() => this.layout()).observe(video)
    video.addEventListener('loadedmetadata', () => this.layout())
    video.addEventListener('resize', () => this.layout())
  }

  /** Canvas pixels = the video's intrinsic size (landmarks are normalized to it); CSS box = the displayed picture. */
  layout(): void {
    const v = this.video
    const vw = v.videoWidth
    const vh = v.videoHeight
    if (!vw || !vh) return
    if (this.canvas.width !== vw || this.canvas.height !== vh) {
      this.canvas.width = vw
      this.canvas.height = vh
    }
    const bw = v.clientWidth
    const bh = v.clientHeight
    const s = Math.min(bw / vw, bh / vh)
    const w = vw * s
    const h = vh * s
    Object.assign(this.canvas.style, {
      left: `${v.offsetLeft + (bw - w) / 2}px`,
      top: `${v.offsetTop + (bh - h) / 2}px`,
      width: `${w}px`,
      height: `${h}px`,
    })
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
  }

  /** Draw raw (not aspect-corrected) landmarks: they are normalized to this video's frame. */
  draw(f: KaggleFrame | null, layers: OverlayLayers): void {
    this.layout()
    this.clear()
    if (!f || (!layers.points && !layers.face)) return
    const ctx = this.ctx
    const w = Math.max(2, this.canvas.width / 400)
    if (layers.face) {
      const face = css('--lm-face', '#b8c7d9')
      ctx.globalAlpha = 0.5
      this.edges(f, OFFSETS.face, FaceLandmarker.FACE_LANDMARKS_LIPS, face, Math.max(1, w / 2))
      ctx.globalAlpha = 0.25
      this.dots(f, OFFSETS.face, 468, face, (i) => !USED.has(OFFSETS.face + i), false)
      ctx.globalAlpha = 0.45
      this.dots(f, OFFSETS.face, 468, face, (i) => USED.has(OFFSETS.face + i), false)
      ctx.globalAlpha = 1
    }
    if (layers.points) {
      const pose = css('--lm-pose', '#aaaaaa')
      const left = css('--lm-left', '#ff9a3c')
      const right = css('--lm-right', '#5aa9ff')
      ctx.globalAlpha = 0.8
      this.edges(f, OFFSETS.pose, POSE_EDGES, pose, w)
      this.dots(f, OFFSETS.pose, 33, pose, (i) => UPPER_POSE.has(i))
      ctx.globalAlpha = 1
      this.edges(f, OFFSETS.left_hand, HandLandmarker.HAND_CONNECTIONS, left, w)
      this.dots(f, OFFSETS.left_hand, 21, left)
      this.edges(f, OFFSETS.right_hand, HandLandmarker.HAND_CONNECTIONS, right, w)
      this.dots(f, OFFSETS.right_hand, 21, right)
    }
  }

  private pt(f: KaggleFrame, g: number): [number, number] | null {
    const x = f[g * C]
    return Number.isNaN(x) ? null : [x * this.canvas.width, f[g * C + 1] * this.canvas.height]
  }

  private edges(f: KaggleFrame, offset: number, conns: Connection[], color: string, width: number) {
    const ctx = this.ctx
    ctx.strokeStyle = color
    ctx.lineWidth = width
    ctx.beginPath()
    for (const { start, end } of conns) {
      const a = this.pt(f, offset + start)
      const b = this.pt(f, offset + end)
      if (a && b) {
        ctx.moveTo(a[0], a[1])
        ctx.lineTo(b[0], b[1])
      }
    }
    ctx.stroke()
  }

  private dots(f: KaggleFrame, from: number, count: number, color: string, only?: (i: number) => boolean, enlargeUsed = true) {
    const ctx = this.ctx
    const r = Math.max(2, this.canvas.width / 320)
    ctx.fillStyle = color
    for (let i = 0; i < count; i++) {
      if (only && !only(i)) continue
      const p = this.pt(f, from + i)
      if (!p) continue
      const size = enlargeUsed && USED.has(from + i) ? r * 1.6 : r * 0.6
      ctx.beginPath()
      ctx.arc(p[0], p[1], size, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}

/** Nearest frame at time t (seconds) in frames sorted by t; null outside [first - slack, last + slack]. */
export function frameAt<T extends { t: number }>(frames: T[], t: number, slack = 0.1): T | null {
  if (!frames.length || t < frames[0].t - slack || t > frames[frames.length - 1].t + slack) return null
  let lo = 0
  let hi = frames.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].t < t) lo = mid + 1
    else hi = mid
  }
  return lo > 0 && Math.abs(frames[lo - 1].t - t) < Math.abs(frames[lo].t - t) ? frames[lo - 1] : frames[lo]
}
