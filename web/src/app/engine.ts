// Lazily loaded, shared pieces: the sign model, MediaPipe, dictionary.json and similar.json.
// Loading starts the first time a screen needs them; a failed load can be retried (loadEngine() again).
// The big files (about 38 MB) are downloaded here with progress; then the model and the three trackers are set
// up one at a time, which keeps peak memory lower on phones.
import { LandmarkExtractor } from '../landmarks'
import { memoryInfo, SignModel } from '../model'
import { diag } from './diag'

const once = <T>(load: () => Promise<T>) => {
  let p: Promise<T> | null = null
  return () =>
    (p ??= load().catch((e) => {
      p = null
      throw e
    }))
}

const base = import.meta.env.BASE_URL

// ---------------------------------------------------------------- loading state, for progress and errors
export interface LoadState {
  phase: 'idle' | 'loading' | 'ready' | 'failed'
  /** 0 to 1 */
  fraction: number
  /** What is happening now, e.g. "Downloading", "Starting hand tracking". */
  step: string
  error?: string
}
let state: LoadState = { phase: 'idle', fraction: 0, step: '' }
const listeners = new Set<(s: LoadState) => void>()
/** Called now and on every change. */
export function onLoadState(f: (s: LoadState) => void): void {
  listeners.add(f)
  f(state)
}
export const loadState = () => state
function update(p: Partial<LoadState>) {
  state = { ...state, ...p }
  diag.set('Loading', 'State', `${state.phase} ${Math.round(state.fraction * 100)}% ${state.step}${state.error ? ` – ${state.error}` : ''}`)
  listeners.forEach((f) => f(state))
}

// ---------------------------------------------------------------- downloads with progress
type AssetKey = 'model' | 'hand' | 'pose' | 'face'
/** Sizes used for the progress bar until the server's Content-Length arrives. */
const ASSETS: Record<AssetKey, { url: string; size: number }> = {
  model: { url: `${base}model/model.onnx`, size: 17_195_631 },
  hand: { url: `${base}mediapipe/hand_landmarker.task`, size: 7_819_105 },
  pose: { url: `${base}mediapipe/pose_landmarker_full.task`, size: 9_398_198 },
  face: { url: `${base}mediapipe/face_landmarker.task`, size: 3_758_596 },
}
const loaded: Record<AssetKey, number> = { model: 0, hand: 0, pose: 0, face: 0 }
const total: Record<AssetKey, number> = { model: ASSETS.model.size, hand: ASSETS.hand.size, pose: ASSETS.pose.size, face: ASSETS.face.size }
/** Set-up steps after the downloads: model, hand, pose, face, test detection. */
const SETUP_STEPS = 5
let stepsDone = 0
const STALL_MS = 30_000

function progress(step?: string) {
  const sum = (r: Record<AssetKey, number>) => Object.values(r).reduce((a, b) => a + b, 0)
  const bytes = Math.min(1, sum(loaded) / sum(total))
  update({ fraction: 0.8 * bytes + 0.2 * Math.min(1, stepsDone / SETUP_STEPS), ...(step ? { step } : {}) })
}

const downloads = new Map<AssetKey, Promise<Uint8Array>>()
/** One download per file, kept for a retry if setting it up fails, dropped once it is in use. */
function download(key: AssetKey): Promise<Uint8Array> {
  let p = downloads.get(key)
  if (!p) {
    p = fetchWithProgress(key)
    p.catch(() => downloads.delete(key))
    downloads.set(key, p)
  }
  return p
}

async function fetchWithProgress(key: AssetKey): Promise<Uint8Array> {
  const ctrl = new AbortController()
  let last = Date.now()
  const watchdog = setInterval(() => Date.now() - last > STALL_MS && ctrl.abort(), 2000)
  try {
    const r = await fetch(ASSETS[key].url, { signal: ctrl.signal })
    if (!r.ok) throw new Error(`${ASSETS[key].url.split('/').pop()}: HTTP ${r.status}`)
    const len = Number(r.headers.get('content-length'))
    if (len > 0 && !r.headers.get('content-encoding')) total[key] = len
    if (!r.body) {
      const buf = new Uint8Array(await r.arrayBuffer())
      loaded[key] = total[key] = buf.byteLength
      progress()
      return buf
    }
    const reader = r.body.getReader()
    const chunks: Uint8Array[] = []
    let n = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      n += value.byteLength
      last = Date.now()
      loaded[key] = Math.min(n, total[key])
      progress()
    }
    const buf = new Uint8Array(n)
    let at = 0
    for (const c of chunks) {
      buf.set(c, at)
      at += c.byteLength
    }
    loaded[key] = total[key] = n
    progress()
    diag.set('Loading', `Downloaded ${key}`, `${(n / 1e6).toFixed(1)} MB`)
    return buf
  } catch (e) {
    loaded[key] = 0
    if (ctrl.signal.aborted) throw new Error(`the download of ${ASSETS[key].url.split('/').pop()} stalled`)
    throw e
  } finally {
    clearInterval(watchdog)
  }
}

/** Set-up steps run one at a time (lower peak memory than all at once). */
let chain: Promise<unknown> = Promise.resolve()
function serial<T>(f: () => Promise<T>): Promise<T> {
  const p = chain.then(f, f)
  chain = p.catch(() => {})
  return p
}

function stepDone(step?: string) {
  stepsDone = Math.min(SETUP_STEPS, stepsDone + 1)
  progress(step)
}

export const getModel = once(async () => {
  const bytes = await download('model')
  const m = await serial(async () => {
    progress('Starting the sign model')
    diag.step('sign model')
    return SignModel.load(undefined, bytes)
  })
  downloads.delete('model')
  diag.set('Model', 'Model', `loaded, ${m.labels.length} signs, ${m.length} frames per lookup`)
  diag.set('Model', 'Wasm memory limit', `${memoryInfo.maximumMB} MB${memoryInfo.retries ? ` (after ${memoryInfo.retries} smaller retries)` : ''}`)
  stepDone()
  return m
})

let extractor: Promise<LandmarkExtractor> | null = null
function createExtractor(delegate?: 'CPU'): Promise<LandmarkExtractor> {
  return (async () => {
    const [hand, pose, face] = await Promise.all([download('hand'), download('pose'), download('face')])
    const ex = await serial(() =>
      LandmarkExtractor.create({
        buffers: { hand, pose, face },
        delegate,
        onStep: (s) => {
          if (!s.startsWith('hand')) stepDone()
          diag.step(s)
          progress(`Starting ${s.replace(/ \((GPU|CPU)\)$/, '')}`)
        },
      }),
    )
    for (const k of ['hand', 'pose', 'face'] as const) downloads.delete(k)
    stepDone()
    if (state.phase === 'ready') update({ step: '' }) // a rebuild on the CPU after loading
    diag.set('Hand tracking', 'Delegate', ex.delegate + (LandmarkExtractor.gpuError ? ` (GPU failed: ${LandmarkExtractor.gpuError})` : ''))
    return ex
  })()
}
/** MediaPipe hand, body and face trackers (GPU, or CPU if the GPU fails). */
export function getExtractor(): Promise<LandmarkExtractor> {
  return (extractor ??= createExtractor().catch((e) => {
    extractor = null
    throw e
  }))
}
/** The GPU failed while tracking: rebuild the trackers on the CPU (shared by the webcam and clip screens). */
export async function useCpuExtractor(): Promise<LandmarkExtractor> {
  const old = extractor ? await extractor.catch(() => null) : null
  if (old?.delegate === 'CPU') return old
  try {
    old?.close()
  } catch {
    /* already broken */
  }
  LandmarkExtractor.gpuError ||= 'failed while tracking'
  stepsDone = SETUP_STEPS - 4
  extractor = createExtractor('CPU').catch((e) => {
    extractor = null
    throw e
  })
  return extractor
}

/** Everything a lookup needs. Safe to call repeatedly; after a failure it tries again. */
export async function loadEngine(): Promise<[LandmarkExtractor, SignModel]> {
  if (state.phase !== 'ready') update({ phase: 'loading', error: undefined })
  if (state.phase === 'loading' && !state.step) progress('Downloading')
  try {
    const [ex, m] = await Promise.all([getExtractor(), getModel(), loadReferenceData()])
    update({ phase: 'ready', fraction: 1, step: '' })
    diag.step('')
    return [ex, m]
  } catch (e) {
    const message = (e as Error)?.message ?? String(e)
    diag.error(`loading failed: ${message}`)
    update({ phase: 'failed', error: message })
    throw e
  }
}

export interface DictionaryEntry {
  display: string | null
  meaning: string | null
  link: string | null
  clip: string | null
  clip_captions: string | null
}
type Similar = Record<string, { sign: string; score: number }[]>

const fetchJson = async <T>(url: string, fallback: T): Promise<T> => {
  try {
    const r = await fetch(url)
    return r.ok ? ((await r.json()) as T) : fallback
  } catch {
    return fallback
  }
}

let dictionary: Record<string, Partial<DictionaryEntry>> = {}
let similar: Similar = {}
export const loadReferenceData = once(async () => {
  const [d, s] = await Promise.all([
    fetchJson<{ signs?: Record<string, Partial<DictionaryEntry>> }>(`${base}dictionary.json`, {}),
    fetchJson<Similar>(`${base}model/similar.json`, {}),
  ])
  dictionary = d.signs ?? {}
  similar = s
})

export function entry(label: string): DictionaryEntry {
  const e = dictionary[label] ?? {}
  return {
    display: e.display || label.charAt(0).toUpperCase() + label.slice(1),
    meaning: e.meaning || null,
    link: e.link || `https://www.signasl.org/sign/${encodeURIComponent(label.toLowerCase())}`,
    clip: e.clip || null,
    clip_captions: e.clip_captions || null,
  }
}

/** Every sign label in dictionary.json (all the model's signs once loaded). */
export const knownLabels = (): string[] => Object.keys(dictionary)

/** "Watch a real signer" link: visible text plus an accessible label naming the site. */
export function signerLink(e: DictionaryEntry): { href: string; aria: string } {
  let site = 'a dictionary'
  try {
    site = new URL(e.link!, location.href).host.replace(/^www\./, '')
  } catch {
    /* keep the generic name */
  }
  return { href: e.link!, aria: `Watch a real signer sign ${e.display} on ${site} (opens in a new tab)` }
}

/** Signs most often confused with this one (from the validation set), as display names. */
export function similarTo(label: string, k = 3): { label: string; display: string }[] {
  return (similar[label] ?? [])
    .filter((s) => s.score > 0)
    .slice(0, k)
    .map((s) => ({ label: s.sign, display: entry(s.sign).display! }))
}
