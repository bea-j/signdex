// Messages between the service worker (sw), the landmark pipeline (offscreen document or the capture iframe),
// the side panel, and content scripts. chrome.runtime.sendMessage reaches every extension context, so each
// message names its target. Content scripts get messages via chrome.tabs.sendMessage (field `cs`).

export type CapturePath = 'video-element' | 'tab-capture'

export interface Thumb {
  /** Capture time, ms (performance.now() in the pipeline context). */
  t: number
  dataUrl: string
}

/** Crop rect as fractions of the top-level viewport (DPR, zoom and capture scaling cancel out). */
export interface Crop {
  x: number
  y: number
  w: number
  h: number
}

export interface Viewport {
  width: number // CSS px (innerWidth / innerHeight)
  height: number
  dpr: number
}

export interface CaptureStatus {
  capturing: boolean
  path?: CapturePath
  reason?: string
  delegate?: 'GPU' | 'CPU'
  /** Size of the frames fed to MediaPipe (native video size, or the cropped tab region). */
  sourceWidth?: number
  sourceHeight?: number
  /** Tab capture only: full captured frame size and crop. */
  tabWidth?: number
  tabHeight?: number
  crop?: Crop | null
  landmarkFps?: number
  thumbFps?: number
  bufferSeconds?: number
  bufferFrames?: number
  lastDetected?: { left: boolean; right: boolean; face: boolean; pose: boolean }
  error?: string
}

export interface ProbeResult {
  frameUrl: string
  isTopFrame: boolean
  videos: number
  chosen?: { width: number; height: number; src: string; paused: boolean; readyState: number }
  /** On-screen rect of the chosen video (CSS px, relative to this frame's viewport). */
  rect?: { x: number; y: number; width: number; height: number }
  viewport: Viewport
  /** true: getImageData threw SecurityError. null: nothing to draw. */
  tainted: boolean | null
  /** Drawn frame is (almost) all black, e.g. DRM-protected video. */
  blank?: boolean
  thumb?: string
  error?: string
}

export type Msg =
  | { target: 'offscreen'; type: 'start'; streamId: string; crop: Crop | null; viewport: Viewport | null; reason: string }
  | { target: 'pipeline'; type: 'stop' }
  | { target: 'pipeline'; type: 'get-buffer' }
  | { target: 'pipeline'; type: 'get-landmarks' }
  | { target: 'pipeline'; type: 'crop'; crop: Crop | null }
  | { target: 'sw'; type: 'probe' }
  | { target: 'sw'; type: 'stop' }
  | { target: 'sw'; type: 'video-path-failed'; reason: string }
  | { target: 'panel'; type: 'thumb'; thumb: Thumb }
  | { target: 'panel'; type: 'status'; status: CaptureStatus }
  | { target: 'panel'; type: 'probe-result'; result: ProbeResult }
  | { target: 'panel'; type: 'log'; text: string }
  /** Alt+Shift+S pressed again while capturing this tab: look up the last few seconds. */
  | { target: 'panel'; type: 'lookup' }

/** Service worker -> content script (chrome.tabs.sendMessage). */
export type CsMsg =
  | { cs: 'start-video'; nonce: string; frameUrl: string }
  | { cs: 'track-rect' }
  | { cs: 'stop' }

/** Content script <-> capture iframe, over a private MessagePort. */
export type PortMsg =
  | { type: 'frame'; bitmap: ImageBitmap; t: number; width: number; height: number }
  | { type: 'ready' }
  | { type: 'done' }
  | { type: 'stop' }
  | { type: 'error'; error: string }

/** sendMessage that ignores "no receiver" (e.g. the side panel is closed). */
export async function send<T = unknown>(msg: Msg): Promise<T | undefined> {
  try {
    return (await chrome.runtime.sendMessage(msg)) as T
  } catch {
    return undefined
  }
}

export const LANDMARK_FPS = 20
export const LANDMARK_BUFFER_SECONDS = 6
export const THUMB_FPS = 10
export const THUMB_BUFFER_SECONDS = 5
export const THUMB_WIDTH = 160
