// On-screen diagnostics for phones, where there is no console: add ?debug=1 to the URL to show a panel with the
// camera size, hand-tracking delegate, model loading and any errors. Values are collected only when it's on.
// One thing is always kept, in localStorage: the loading step in progress. If iOS kills the tab while loading
// (memory), the reloaded page's debug panel can say where the previous visit stopped.

const params = new URLSearchParams(location.search)
const enabled = params.has('debug') && params.get('debug') !== '0'
const STEP_KEY = 'signdex.loadStep'

const values = new Map<string, Map<string, string>>()
const errors: string[] = []
const since = performance.now()
const stamp = () => `${((performance.now() - since) / 1000).toFixed(1)}s`

let previousStep: string | null = null
try {
  previousStep = localStorage.getItem(STEP_KEY)
} catch {
  /* storage blocked */
}

export const diag = {
  enabled,
  set(group: string, key: string, value: string | number | boolean | null | undefined) {
    if (!enabled) return
    if (!values.has(group)) values.set(group, new Map())
    values.get(group)!.set(key, value === null || value === undefined || value === '' ? '–' : String(value))
    queue()
  },
  error(message: string) {
    if (!enabled) return
    errors.push(`${stamp()} ${message}`.slice(0, 600))
    if (errors.length > 40) errors.shift()
    queue()
  },
  /** The loading step now running ('' when done). Kept across a crash of the tab. */
  step(step: string) {
    try {
      if (step) localStorage.setItem(STEP_KEY, `${step} (${new Date().toLocaleTimeString()})`)
      else localStorage.removeItem(STEP_KEY)
    } catch {
      /* storage blocked */
    }
    diag.set('Loading', 'Step', step || 'done')
  },
}

// ---------------------------------------------------------------- panel
let panel: HTMLElement | null = null
let body: HTMLElement | null = null
let queued = false
function queue() {
  if (!body || queued || panel?.hidden) return
  queued = true
  setTimeout(() => {
    queued = false
    render()
  }, 300)
}

/** What a phone screen shows first: the state of things, then errors, then the device details. */
const ORDER = ['Loading', 'Camera', 'Hand tracking', 'Model', 'Last lookup', 'Clip']

function text(): string {
  const out: string[] = []
  const group = (name: string) => {
    const kv = values.get(name)
    if (!kv) return
    out.push(`[${name}]`)
    for (const [k, v] of kv) out.push(`  ${k}: ${v}`)
  }
  ORDER.forEach(group)
  out.push(`[Errors and warnings] ${errors.length ? '' : 'none'}`)
  for (const e of errors) out.push(`  ${e}`)
  for (const name of values.keys()) if (!ORDER.includes(name)) group(name)
  return out.join('\n')
}

function render() {
  if (body) body.textContent = text()
}

function device() {
  const s = screen as Screen & { orientation?: ScreenOrientation }
  diag.set('Device', 'Browser', navigator.userAgent)
  diag.set('Device', 'Viewport', `${innerWidth} × ${innerHeight} CSS px, DPR ${devicePixelRatio}, ${s.orientation?.type ?? ''}`)
  diag.set('Device', 'Touch', `${matchMedia('(pointer: coarse)').matches ? 'coarse pointer' : 'fine pointer'}, ${navigator.maxTouchPoints} touch points`)
  diag.set('Device', 'Memory / cores', `${(navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? '?'} GB, ${navigator.hardwareConcurrency ?? '?'} cores`)
  diag.set('Device', 'WebAssembly', `${typeof WebAssembly === 'object' ? 'yes' : 'no'}; SharedArrayBuffer ${typeof SharedArrayBuffer !== 'undefined' ? 'yes' : 'no'}; cross-origin isolated ${self.crossOriginIsolated ? 'yes' : 'no'}`)
  diag.set('Device', 'OffscreenCanvas', typeof OffscreenCanvas !== 'undefined' ? 'yes' : 'no')
  diag.set('Device', 'MediaRecorder', typeof MediaRecorder !== 'undefined' ? 'yes' : 'no')
  diag.set('Device', 'Camera API', 'mediaDevices' in navigator && !!navigator.mediaDevices ? 'yes' : 'no (needs HTTPS)')
}

function mount() {
  const toggle = document.createElement('button')
  toggle.type = 'button'
  toggle.className = 'debug-toggle'
  toggle.textContent = 'Debug'
  toggle.setAttribute('aria-expanded', 'false')
  toggle.setAttribute('aria-controls', 'debug-panel')
  panel = document.createElement('section')
  panel.id = 'debug-panel'
  panel.className = 'debug-panel'
  panel.setAttribute('aria-label', 'Debug information')
  panel.hidden = true
  const bar = document.createElement('div')
  bar.className = 'debug-bar'
  const copy = Object.assign(document.createElement('button'), { type: 'button', textContent: 'Copy' })
  const close = Object.assign(document.createElement('button'), { type: 'button', textContent: 'Close' })
  bar.append(Object.assign(document.createElement('strong'), { textContent: 'Signdex debug' }), copy, close)
  body = document.createElement('pre')
  panel.append(bar, body)
  document.body.append(toggle, panel)
  const show = (open: boolean) => {
    panel!.hidden = !open
    toggle.hidden = open
    toggle.setAttribute('aria-expanded', String(open))
    if (open) {
      device()
      render()
    }
  }
  toggle.addEventListener('click', () => show(true))
  close.addEventListener('click', () => {
    show(false)
    toggle.focus()
  })
  copy.addEventListener('click', () => {
    void navigator.clipboard?.writeText(text()).then(
      () => (copy.textContent = 'Copied'),
      () => (copy.textContent = 'Copy failed'),
    )
    setTimeout(() => (copy.textContent = 'Copy'), 2000)
  })
  setInterval(() => panel && !panel.hidden && render(), 1000)
  show(true)
}

if (enabled) {
  if (previousStep) diag.set('Loading', 'Previous visit stopped while loading', previousStep)
  device()
  addEventListener('error', (e) => diag.error(`error: ${e.message} ${e.filename ? `(${e.filename.split('/').pop()}:${e.lineno})` : ''}`))
  addEventListener('unhandledrejection', (e) => diag.error(`unhandled: ${(e.reason as Error)?.message ?? String(e.reason)}`))
  // copy console errors and warnings into the panel (they still go to the console as usual)
  for (const level of ['error', 'warn'] as const) {
    const original = console[level].bind(console)
    console[level] = (...args: unknown[]) => {
      original(...args)
      diag.error(`console.${level}: ${args.map((a) => (a instanceof Error ? a.message : typeof a === 'string' ? a : JSON.stringify(a))).join(' ')}`)
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount)
  else mount()
}
