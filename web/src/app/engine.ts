// Lazily loaded, shared pieces: the sign model, MediaPipe, dictionary.json and similar.json.
// Loading starts the first time a screen needs them; a failed load can be retried.
import { LandmarkExtractor } from '../landmarks'
import { SignModel } from '../model'

const once = <T>(load: () => Promise<T>) => {
  let p: Promise<T> | null = null
  return () =>
    (p ??= load().catch((e) => {
      p = null
      throw e
    }))
}

export const getModel = once(() => SignModel.load())
export const getExtractor = once(() => LandmarkExtractor.create())

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

const base = import.meta.env.BASE_URL
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
