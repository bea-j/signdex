// Draws a reference clip as a skeleton on a square canvas: hand bones as lines, joints as dots, upper-body pose
// and a faint lip outline, in the design's colors. Mirrored by default, so it matches the user's mirrored webcam
// view (the signer's dominant right hand appears on the right).
import { FaceLandmarker, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision'
import type { RefClip } from '../reference'

const HAND_BASE = { left_hand: 468, right_hand: 522 } as const
const POSE_BASE = 489
const UPPER_POSE = new Set([0, 11, 12, 13, 14, 15, 16])
const css = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback

type Edge = [number, number]

export interface SkeletonOptions {
  mirror?: boolean
  /** Pause on the last frame before looping, ms. */
  holdMs?: number
}

export class SkeletonPlayer {
  private readonly ctx: CanvasRenderingContext2D
  private readonly hands: { edges: Edge[]; joints: number[]; color: string }[] = []
  private readonly poseEdges: Edge[] = []
  private readonly lipEdges: Edge[] = []
  private nose = -1
  private scale = 1
  private ox = 0
  private oy = 0
  private frame = 0
  private raf = 0
  private due = 0
  private playing = false
  /** 0..1: brighten and glow the hands (home page demo). */
  highlight = 0
  onFrame?: (frame: number, count: number) => void

  constructor(
    private readonly canvas: HTMLCanvasElement,
    readonly clip: RefClip,
    private readonly opts: SkeletonOptions = {},
  ) {
    this.ctx = canvas.getContext('2d')!
    const pos = new Map<number, number>()
    clip.points.forEach((p, i) => pos.set(p.index, i))
    const map = (edges: { start: number; end: number }[], base: number): Edge[] =>
      edges
        .map(({ start, end }) => [pos.get(base + start), pos.get(base + end)] as const)
        .filter((e): e is [number, number] => e[0] !== undefined && e[1] !== undefined)
    for (const g of ['left_hand', 'right_hand'] as const) {
      this.hands.push({
        edges: map(HandLandmarker.HAND_CONNECTIONS, HAND_BASE[g]),
        joints: clip.points.flatMap((p, i) => (p.group === g ? [i] : [])),
        color: g === 'right_hand' ? css('--ref-hand', '#cfe0d1') : css('--ref-hand-2', '#dbc99f'),
      })
    }
    this.poseEdges.push(
      ...map(
        PoseLandmarker.POSE_CONNECTIONS.filter((c) => UPPER_POSE.has(c.start) && UPPER_POSE.has(c.end)),
        POSE_BASE,
      ),
    )
    this.lipEdges.push(...map(FaceLandmarker.FACE_LANDMARKS_LIPS, 0))
    this.nose = pos.get(POSE_BASE) ?? -1
    this.fit()
    this.frame = this.bestFrame
  }

  /**
   * Fixed body-centred view, the same for every sign: x in [-1.5, 1.5] and y in [-1.75, 1.25] shoulder widths
   * (0 = mid-shoulder), which keeps the head, shoulders and signing space in frame.
   */
  fit() {
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const W = Math.round((this.canvas.clientWidth || this.canvas.width) * dpr)
    const H = Math.round((this.canvas.clientHeight || this.canvas.height) * dpr)
    this.canvas.width = W
    this.canvas.height = H
    this.scale = Math.min(W / 3, H / 3)
    this.ox = W / 2
    this.oy = H / 2 + 0.25 * this.scale
  }

  private xy(f: Float32Array, i: number): [number, number] | null {
    const x = f[i * 2]
    const y = f[i * 2 + 1]
    if (Number.isNaN(x) || Number.isNaN(y)) return null
    const sx = this.opts.mirror === false ? 1 : -1
    return [x * sx * this.scale + this.ox, y * this.scale + this.oy]
  }

  private lines(f: Float32Array, edges: Edge[], color: string, width: number) {
    const ctx = this.ctx
    ctx.strokeStyle = color
    ctx.lineWidth = width
    ctx.lineCap = 'round'
    ctx.beginPath()
    for (const [a, b] of edges) {
      const p = this.xy(f, a)
      const q = this.xy(f, b)
      if (p && q) {
        ctx.moveTo(p[0], p[1])
        ctx.lineTo(q[0], q[1])
      }
    }
    ctx.stroke()
  }

  draw(i = this.frame) {
    const f = this.clip.frames[Math.max(0, Math.min(this.clip.frames.length - 1, i))]
    const ctx = this.ctx
    const u = Math.min(this.canvas.width, this.canvas.height) / 100 // 1% of the shorter side
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    if (!f) return
    const head = this.nose >= 0 ? this.xy(f, this.nose) : null
    if (head) {
      ctx.strokeStyle = css('--ref-pose', 'rgba(184, 188, 179, 0.55)')
      ctx.lineWidth = Math.max(1, 1.1 * u)
      ctx.beginPath()
      ctx.ellipse(head[0], head[1] - 0.05 * this.scale, 0.33 * this.scale, 0.42 * this.scale, 0, 0, Math.PI * 2)
      ctx.stroke()
    }
    this.lines(f, this.poseEdges, css('--ref-pose', 'rgba(184, 188, 179, 0.55)'), Math.max(1, 1.4 * u))
    this.lines(f, this.lipEdges, css('--ref-lips', 'rgba(200, 197, 212, 0.55)'), Math.max(0.75, 0.6 * u))
    const glow = this.highlight
    for (const h of this.hands) {
      ctx.save()
      if (glow > 0) {
        ctx.shadowColor = h.color
        ctx.shadowBlur = 6 * u * glow
      }
      this.lines(f, h.edges, h.color, Math.max(1, (1.1 + 0.5 * glow) * u))
      ctx.fillStyle = glow > 0 ? '#ffffff' : h.color
      for (const j of h.joints) {
        const p = this.xy(f, j)
        if (!p) continue
        ctx.beginPath()
        ctx.arc(p[0], p[1], Math.max(1.2, (1.25 + 0.6 * glow) * u), 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.restore()
    }
  }

  get isPlaying() {
    return this.playing
  }

  get frameCount() {
    return this.clip.frames.length
  }

  /** The frame with the most visible hand joints (ties: nearest the middle), a good still for the sign. */
  get bestFrame(): number {
    const mid = (this.clip.frames.length - 1) / 2
    let best = 0
    let bestScore = -1
    this.clip.frames.forEach((f, t) => {
      let n = 0
      for (const h of this.hands) for (const j of h.joints) if (!Number.isNaN(f[j * 2])) n++
      const score = n - Math.abs(t - mid) / 1000
      if (score > bestScore) {
        bestScore = score
        best = t
      }
    })
    return best
  }

  /** Show one frame without animating (default: the frame that shows the hands best). */
  still(i = this.bestFrame) {
    this.pause()
    this.frame = i
    this.draw()
  }

  play(from?: number) {
    if (from !== undefined) this.frame = from
    if (this.playing) return
    this.playing = true
    this.due = performance.now()
    const step = 1000 / this.clip.fps
    const tick = (now: number) => {
      if (!this.playing) return
      if (now >= this.due) {
        this.draw()
        this.onFrame?.(this.frame, this.clip.frames.length)
        const last = this.frame >= this.clip.frames.length - 1
        this.frame = last ? 0 : this.frame + 1
        this.due = now + (last ? (this.opts.holdMs ?? 450) : step)
      }
      this.raf = requestAnimationFrame(tick)
    }
    this.raf = requestAnimationFrame(tick)
  }

  pause() {
    this.playing = false
    cancelAnimationFrame(this.raf)
  }
}

export const prefersReducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches
