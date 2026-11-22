// Animated light background: public/bg/signdex-bg.js (the "dapple" p5 sketch, unchanged) on self-hosted p5
// (public/vendor/p5.min.js). Loaded only when switched on; when off, the static ambient background shows.
// It shares the GPU with on-device MediaPipe, so it is paused whenever tracking or a lookup runs (bgBusy).
// "Background" switch in the header, remembered; default off on phones and low-powered devices.
interface SignDexBg {
  pause(): void
  resume(): void
  set(key: string, value: unknown): void
  settings: Record<string, unknown>
}
declare global {
  interface Window {
    signdexBg?: SignDexBg
  }
}

const KEY = 'signdex.background'
const base = import.meta.env.BASE_URL
const toggle = document.getElementById('bg-toggle') as HTMLInputElement

/** Phones, touch-first devices, and machines that look low-powered start with the background off. */
export function backgroundDefault(): boolean {
  const nav = navigator as Navigator & { deviceMemory?: number; connection?: { saveData?: boolean } }
  const phone = matchMedia('(max-width: 760px), (pointer: coarse)').matches
  const lowPower = (nav.hardwareConcurrency ?? 8) <= 4 || (nav.deviceMemory ?? 8) <= 4 || !!nav.connection?.saveData
  return !phone && !lowPower
}

function stored(): boolean | null {
  try {
    const v = localStorage.getItem(KEY)
    return v === null ? null : v === 'on'
  } catch {
    return null
  }
}

let enabled = stored() ?? backgroundDefault()
let busy = 0
let loading: Promise<void> | null = null

const loadScript = (src: string) =>
  new Promise<void>((resolve, reject) => {
    const s = document.createElement('script')
    s.src = src
    s.onload = () => resolve()
    s.onerror = () => reject(new Error(`could not load ${src}`))
    document.body.append(s)
  })

function ensureLoaded(): Promise<void> {
  // p5 first, then the sketch (it calls new p5(...) from the global); only ever once
  return (loading ??= loadScript(`${base}vendor/p5.min.js`).then(() => loadScript(`${base}bg/signdex-bg.js`)))
}

function apply() {
  document.body.classList.toggle('bg-live', enabled && !!window.signdexBg)
  const canvas = document.getElementById('signdex-bg')
  if (canvas) canvas.style.display = enabled ? '' : 'none'
  const bg = window.signdexBg
  if (!bg) return
  if (enabled && busy === 0) bg.resume()
  else bg.pause()
}

export async function setBackground(on: boolean, remember = true) {
  enabled = on
  toggle.checked = on
  if (remember) {
    try {
      localStorage.setItem(KEY, on ? 'on' : 'off')
    } catch {
      /* this visit only */
    }
  }
  if (on) {
    try {
      await ensureLoaded()
    } catch (e) {
      console.warn('background unavailable:', e)
      enabled = false
      toggle.checked = false
    }
  }
  apply()
}

/** Mark on-device ML as running (true) or finished (false). The background pauses while anything is busy. */
export function bgBusy(on: boolean) {
  busy = Math.max(0, busy + (on ? 1 : -1))
  apply()
}

toggle.addEventListener('change', () => void setBackground(toggle.checked))
void setBackground(enabled, false)
