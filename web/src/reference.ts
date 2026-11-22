// Reference animations: one representative PopSign ASL training sequence per sign (model/reference.py), as
// features stage-1 landmarks (trimmed, left-dominant mirrored, centered on mid-shoulder, / shoulder width).
// public/reference/signs.bin is loaded lazily: the header first, then only the bytes of each sign that is shown
// (HTTP range requests). Where ranges are not supported (e.g. extension files), the whole file is fetched once.

export interface RefPoint {
  group: 'left_hand' | 'right_hand' | 'lips' | 'pose'
  /** Kaggle landmark index (0-542). */
  index: number
}

export interface RefClip {
  label: string
  /** Per frame: x, y for each point (NaN = missing), in shoulder widths; y already scaled to the video aspect. */
  frames: Float32Array[]
  points: RefPoint[]
  fps: number
  /** The reference model's probability for this sign on this sequence. */
  confidence: number
}

interface Header {
  format: string
  scale: number
  aspect_y: number
  fps: number
  points: [RefPoint['group'], number][]
  signs: { label: string; offset: number; frames: number; confidence: number }[]
}

const URL_ = `${import.meta.env.BASE_URL}reference/signs.bin`
let headerP: Promise<{ header: Header; dataStart: number } | null> | null = null
let whole: ArrayBuffer | null = null // set when the server ignored a range request, or after a full preload
let wholeLoading: Promise<void> | null = null
const cache = new Map<string, Promise<RefClip | null>>()

/** Bytes [start, end) of the file; falls back to the whole file when ranges are not honored. */
async function bytes(start: number, end: number): Promise<Uint8Array> {
  if (wholeLoading) await wholeLoading // a full download is under way: use it instead of extra range requests
  if (whole) return new Uint8Array(whole, start, end - start)
  const r = await fetch(URL_, { headers: { Range: `bytes=${start}-${end - 1}` } })
  if (!r.ok) throw new Error(`reference: HTTP ${r.status}`)
  const buf = await r.arrayBuffer()
  if (r.status === 206) return new Uint8Array(buf)
  whole = buf
  return new Uint8Array(whole, start, end - start)
}

function header() {
  return (headerP ??= (async () => {
    const head = await bytes(0, 8)
    if (new TextDecoder().decode(head.subarray(0, 4)) !== 'SDXR') return null
    const len = new DataView(head.buffer, head.byteOffset, 8).getUint32(4, true)
    const json = await bytes(8, 8 + len)
    return { header: JSON.parse(new TextDecoder().decode(json)) as Header, dataStart: 8 + len }
  })().catch(() => null))
}

/** Fetch the whole file once (about 1 MB), for pages that show many signs; later clips decode from memory. */
export function preloadAllReferences(): Promise<void> {
  return (wholeLoading ??= fetch(URL_)
    .then(async (r) => {
      if (r.ok) whole = await r.arrayBuffer()
    })
    .catch(() => {}))
}

/** The reference clip for a sign label, or null if there is none (file missing or sign not in it). */
export function referenceClip(label: string): Promise<RefClip | null> {
  let p = cache.get(label)
  if (!p) {
    p = (async () => {
      const h = await header()
      const s = h?.header.signs.find((x) => x.label === label)
      if (!h || !s) return null
      const P = h.header.points.length
      const { scale, aspect_y: ay } = h.header
      const raw = await bytes(h.dataStart + s.offset, h.dataStart + s.offset + s.frames * P * 2)
      const q = new Int8Array(raw.buffer, raw.byteOffset, raw.byteLength)
      const frames: Float32Array[] = []
      for (let t = 0; t < s.frames; t++) {
        const f = new Float32Array(P * 2)
        for (let p2 = 0; p2 < P; p2++) {
          const i = (t * P + p2) * 2
          f[p2 * 2] = q[i] === -128 ? Number.NaN : q[i] * scale
          f[p2 * 2 + 1] = q[i + 1] === -128 ? Number.NaN : q[i + 1] * scale * ay
        }
        frames.push(f)
      }
      return {
        label,
        frames,
        points: h.header.points.map(([group, index]) => ({ group, index })),
        fps: h.header.fps,
        confidence: s.confidence,
      } satisfies RefClip
    })().catch(() => null)
    cache.set(label, p)
  }
  return p
}

/** Signs in the reference file with the model's confidence on their sequence (empty if there is no file). */
export async function referenceSigns(): Promise<{ label: string; confidence: number }[]> {
  const h = await header()
  return h ? h.header.signs.map(({ label, confidence }) => ({ label, confidence })) : []
}
