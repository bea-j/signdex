// Shared landmark pipeline for both capture paths (runs in the capture iframe or the offscreen document).
// Frames in -> MediaPipe landmarks at ~20 fps -> rolling 6 s landmark buffer + 5 s of small thumbnails.
import {
  COORDS_PER_POINT as C,
  LandmarkExtractor,
  OFFSETS,
  toRecording,
  type KaggleFrame,
  type LandmarkRecording,
} from '@shared/landmarks'
import {
  LANDMARK_BUFFER_SECONDS,
  LANDMARK_FPS,
  send,
  THUMB_BUFFER_SECONDS,
  THUMB_FPS,
  THUMB_WIDTH,
  type CapturePath,
  type CaptureStatus,
  type Thumb,
} from './messages'

interface Entry {
  wall: number // ms
  t: number // seconds since capture start
  frame: KaggleFrame
  width: number
  height: number
}

const present = (f: KaggleFrame, offset: number, n: number) => {
  for (let i = 0; i < n; i++) if (!Number.isNaN(f[(offset + i) * C])) return true
  return false
}

export class LandmarkPipeline {
  status: CaptureStatus
  private ex?: LandmarkExtractor
  private entries: Entry[] = []
  private thumbs: Thumb[] = []
  private due = 0
  private thumbDue = 0
  private readonly start = performance.now()
  private canvas: OffscreenCanvas | null = null
  private heartbeat: ReturnType<typeof setInterval>

  constructor(path: CapturePath, reason: string) {
    this.status = { capturing: false, path, reason }
    this.heartbeat = setInterval(() => this.report(), 1000)
  }

  async init(): Promise<void> {
    this.ex = await LandmarkExtractor.create({ assetsBase: chrome.runtime.getURL('mediapipe') })
    this.status.delegate = this.ex.delegate
    this.status.capturing = true
    this.report()
  }

  /** Cheap check so callers can skip cropping/drawing frames the pipeline would drop anyway. */
  wantsFrame(now = performance.now()): boolean {
    return !!this.ex && now >= this.due
  }

  /**
   * Feed one frame. Everything that reads `src` happens synchronously before the first await,
   * so the caller may close an ImageBitmap/VideoFrame right after calling (then await the promise).
   */
  push(src: TexImageSource & CanvasImageSource, width: number, height: number): Promise<void> {
    const now = performance.now()
    if (!this.ex || now < this.due) return Promise.resolve()
    const interval = 1000 / LANDMARK_FPS
    this.due = this.due < now - interval ? now + interval : this.due + interval

    const frame = this.ex.detect(src, now)
    this.entries.push({ wall: now, t: (now - this.start) / 1000, frame, width, height })
    while (this.entries.length && this.entries[0].wall < now - LANDMARK_BUFFER_SECONDS * 1000) this.entries.shift()
    this.status.sourceWidth = width
    this.status.sourceHeight = height
    this.status.lastDetected = {
      left: present(frame, OFFSETS.left_hand, 21),
      right: present(frame, OFFSETS.right_hand, 21),
      face: present(frame, OFFSETS.face, 468),
      pose: present(frame, OFFSETS.pose, 33),
    }

    if (now < this.thumbDue) return Promise.resolve()
    this.thumbDue = now + 1000 / THUMB_FPS - 5
    return this.thumb(src, width, height, frame, now)
  }

  private thumb(src: CanvasImageSource, width: number, height: number, frame: KaggleFrame, now: number): Promise<void> {
    const tw = THUMB_WIDTH
    const th = Math.max(1, Math.round((THUMB_WIDTH * height) / width))
    if (!this.canvas || this.canvas.width !== tw || this.canvas.height !== th) this.canvas = new OffscreenCanvas(tw, th)
    const ctx = this.canvas.getContext('2d')!
    ctx.drawImage(src, 0, 0, tw, th)
    // hand dots (left slot orange, right slot blue) + shoulders, so the strip shows landmarks are running
    const dot = (offset: number, n: number, color: string) => {
      ctx.fillStyle = color
      for (let i = 0; i < n; i++) {
        const x = frame[(offset + i) * C]
        if (Number.isNaN(x)) continue
        ctx.fillRect(x * tw - 1, frame[(offset + i) * C + 1] * th - 1, 2, 2)
      }
    }
    dot(OFFSETS.left_hand, 21, '#ff9a3c')
    dot(OFFSETS.right_hand, 21, '#5aa9ff')
    dot(OFFSETS.pose + 11, 2, '#dddddd')
    return this.canvas.convertToBlob({ type: 'image/jpeg', quality: 0.6 }).then(
      (blob) =>
        new Promise<void>((resolve) => {
          const r = new FileReader()
          r.onload = () => {
            const thumb = { t: now, dataUrl: r.result as string }
            this.thumbs.push(thumb)
            while (this.thumbs.length && this.thumbs[0].t < now - THUMB_BUFFER_SECONDS * 1000) this.thumbs.shift()
            void send({ target: 'panel', type: 'thumb', thumb })
            resolve()
          }
          r.readAsDataURL(blob)
        }),
    )
  }

  report(): void {
    const now = performance.now()
    if (this.status.capturing) {
      this.status.landmarkFps = this.entries.filter((e) => e.wall > now - 1000).length
      this.status.thumbFps = this.thumbs.filter((t) => t.t > now - 1000).length
      this.status.bufferFrames = this.entries.length
      this.status.bufferSeconds = this.entries.length ? Math.round((now - this.entries[0].wall) / 100) / 10 : 0
    }
    void send({ target: 'panel', type: 'status', status: this.status })
  }

  buffer(): { thumbs: Thumb[]; status: CaptureStatus } {
    return { thumbs: this.thumbs, status: this.status }
  }

  /** Last ~6 s of raw landmarks as a signseek-landmarks-v1 recording (same format as the web debug page). */
  landmarks(): LandmarkRecording | null {
    const last = this.entries[this.entries.length - 1]
    if (!last) return null
    return toRecording(this.entries, last.width, last.height, LANDMARK_FPS, `extension:${this.status.path}`)
  }

  fail(error: string): void {
    this.status.capturing = false
    this.status.error = error
    this.report()
  }

  stop(): void {
    clearInterval(this.heartbeat)
    this.status = { capturing: false, path: this.status.path }
    this.ex?.close()
    this.ex = undefined
    void send({ target: 'panel', type: 'status', status: this.status })
  }
}
