// Offscreen document: FALLBACK capture path. Tab capture stream -> crop to the <video> element's on-screen rect
// -> landmark pipeline. Used only when the page's <video> canvas is tainted or blank (or the primary path failed).
import { type Crop, type Msg, type Viewport } from './messages'
import { LandmarkPipeline } from './pipeline'

let stream: MediaStream | null = null
let pipeline: LandmarkPipeline | null = null
let crop: Crop | null = null
let generation = 0
let canvas: OffscreenCanvas | null = null

function stop(): void {
  generation++
  stream?.getTracks().forEach((t) => t.stop())
  stream = null
  pipeline?.stop()
  pipeline = null
}

/** Crop fractions -> pixel rect of this captured frame. Fractions make DPR, zoom and capture scaling cancel out. */
function cropRect(w: number, h: number): [number, number, number, number] {
  if (!crop) return [0, 0, w, h]
  const sx = Math.round(crop.x * w)
  const sy = Math.round(crop.y * h)
  const sw = Math.max(1, Math.min(w - sx, Math.round(crop.w * w)))
  const sh = Math.max(1, Math.min(h - sy, Math.round(crop.h * h)))
  return [sx, sy, sw, sh]
}

async function read(track: MediaStreamTrack, gen: number, p: LandmarkPipeline): Promise<void> {
  const Processor = (globalThis as any).MediaStreamTrackProcessor // Chrome 94+, not in TS's DOM lib yet
  const reader = new Processor({ track }).readable.getReader() as ReadableStreamDefaultReader<VideoFrame>
  while (gen === generation) {
    const { value: frame, done } = await reader.read()
    if (done || !frame) break
    if (!p.wantsFrame()) {
      frame.close()
      continue
    }
    const fw = frame.displayWidth
    const fh = frame.displayHeight
    const [sx, sy, sw, sh] = cropRect(fw, fh)
    if (!canvas || canvas.width !== sw || canvas.height !== sh) canvas = new OffscreenCanvas(sw, sh)
    canvas.getContext('2d')!.drawImage(frame, sx, sy, sw, sh, 0, 0, sw, sh)
    frame.close()
    p.status.tabWidth = fw
    p.status.tabHeight = fh
    p.status.crop = crop
    await p.push(canvas, sw, sh)
  }
  reader.cancel().catch(() => {})
}

async function start(streamId: string, initialCrop: Crop | null, viewport: Viewport | null, reason: string) {
  stop()
  const gen = ++generation
  crop = initialCrop
  const p = new LandmarkPipeline('tab-capture', reason)
  pipeline = p
  // Ask for the tab at device-pixel resolution so the cropped video region isn't downscaled.
  const maxWidth = Math.min(3840, Math.round((viewport?.width ?? 1280) * (viewport?.dpr ?? 1)))
  const maxHeight = Math.min(2160, Math.round((viewport?.height ?? 720) * (viewport?.dpr ?? 1)))
  try {
    await p.init()
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId, maxWidth, maxHeight, maxFrameRate: 30 } },
    } as unknown as MediaStreamConstraints)
    if (gen !== generation) return
    const track = stream.getVideoTracks()[0]
    track.addEventListener('ended', () => gen === generation && stop())
    void read(track, gen, p)
  } catch (e) {
    p.fail(`tab capture failed: ${(e as Error).message}`)
  }
}

chrome.runtime.onMessage.addListener((msg: Msg, _sender, sendResponse) => {
  if (msg.target === 'offscreen' && msg.type === 'start') {
    void start(msg.streamId, msg.crop, msg.viewport, msg.reason)
    sendResponse(true)
    return
  }
  if (msg.target !== 'pipeline') return
  if (msg.type === 'crop') {
    crop = msg.crop
    return
  }
  if (!pipeline) return // only the active pipeline answers
  if (msg.type === 'stop') {
    stop()
    sendResponse(true)
  } else if (msg.type === 'get-buffer') {
    sendResponse(pipeline.buffer())
  } else if (msg.type === 'get-landmarks') {
    sendResponse(pipeline.landmarks())
  }
})
