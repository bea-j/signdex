// Results screen: the user's clip on a loop, and the top 5 as result cards.
import { LOW_CONFIDENCE, type SignMatch } from '../model'
import { entry, similarTo } from './engine'
import { savedSigns } from './saved'

export interface ResultView {
  matches: SignMatch[]
  source: 'webcam' | 'upload'
  /** Clip to replay: a blob URL (webcam recording) or the uploaded file's URL. */
  clipUrl: string | null
  /** For uploads: the selected window inside the file. */
  start?: number
  end?: number
  seconds: number
  mirror?: boolean
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const HIGH = 0.6

export function level(p: number): { cls: string; label: string } {
  if (p >= HIGH) return { cls: 'conf-high', label: 'High confidence' }
  if (p >= LOW_CONFIDENCE) return { cls: 'conf-amber', label: 'Medium confidence' }
  return { cls: 'conf-low', label: 'Low confidence' }
}

const icon = (name: string, cls = 'icon-sm') => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`)

// ---------------------------------------------------------------- looping clip
let stopLoop: (() => void) | null = null

function playLoop(view: ResultView) {
  stopLoop?.()
  const video = $<HTMLVideoElement>('result-video')
  video.classList.toggle('mirror', !!view.mirror)
  if (!view.clipUrl) {
    video.removeAttribute('src')
    video.load()
    return
  }
  const start = view.start ?? 0
  const end = view.end
  video.src = view.clipUrl
  video.loop = end === undefined
  let alive = true
  const check = () => {
    if (end !== undefined && video.currentTime >= end) video.currentTime = start
  }
  const tick = () => {
    if (!alive) return
    check()
    video.requestVideoFrameCallback(tick)
  }
  const onLoaded = () => {
    video.currentTime = start
    void video.play().catch(() => {})
    if ('requestVideoFrameCallback' in video) tick()
  }
  const onEnded = () => {
    video.currentTime = start
    void video.play().catch(() => {})
  }
  video.addEventListener('loadedmetadata', onLoaded, { once: true })
  video.addEventListener('timeupdate', check) // also covers browsers without requestVideoFrameCallback
  video.addEventListener('ended', onEnded)
  stopLoop = () => {
    alive = false
    video.removeEventListener('loadedmetadata', onLoaded)
    video.removeEventListener('timeupdate', check)
    video.removeEventListener('ended', onEnded)
    video.pause()
  }
}

export function stopResults() {
  stopLoop?.()
}

// ---------------------------------------------------------------- cards
let cardSubscriptions: (() => void)[] = []

function card(m: SignMatch, rank: number, unsure: boolean): HTMLElement {
  const e = entry(m.label)
  const lv = level(m.confidence)
  const pct = Math.round(m.confidence * 1000) / 10
  const pctText = pct >= 10 ? pct.toFixed(0) : pct.toFixed(1)
  const sims = similarTo(m.label)
  const el = document.createElement('article')
  el.className = `result-card glass-panel ${rank === 1 ? 'first' : ''}`
  el.setAttribute('aria-label', `Match ${rank}: ${e.display}, ${pctText} percent`)
  el.innerHTML = `
    <div class="card-row">
      <span class="rank" aria-hidden="true">${rank}</span>
      <div class="card-main">
        <div class="card-top">
          <div>
            <div class="name-row">
              <h2 class="sign-name">${esc(e.display!)}</h2>
              ${rank === 1 ? `<span class="best-tag ${unsure ? 'unsure' : ''}">${unsure ? 'Top guess' : 'Best match'}</span>` : ''}
            </div>
            ${e.meaning ? `<p class="meaning">${esc(e.meaning)}</p>` : ''}
          </div>
          <div class="score ${lv.cls}"><p class="pct">${pctText}%</p><p class="level">${lv.label}</p></div>
        </div>
        <div class="bar ${lv.cls}" aria-hidden="true"><span style="width:${Math.max(1, pct)}%"></span></div>
        <div class="card-bottom">
          <div class="similar">${
            sims.length
              ? `<span class="similar-label">Looks similar to</span>${sims.map((s) => `<span class="chip">${esc(s.display)}</span>`).join('')}`
              : ''
          }</div>
          <div class="card-actions">
            ${e.clip ? `<button class="link" type="button" data-example aria-expanded="false">${icon('play')} Example</button>` : ''}
            <a class="link" href="${esc(e.link!)}" target="_blank" rel="noreferrer" aria-label="See ${esc(e.display!)} in a dictionary (opens in a new tab)">See this sign ${icon('external')}</a>
            <button class="save" type="button" aria-pressed="false">${icon('bookmark')} <span>Save</span></button>
          </div>
        </div>
        <div class="example" hidden></div>
      </div>
    </div>`

  // save toggle
  const save = el.querySelector<HTMLButtonElement>('.save')!
  const sync = () => {
    const on = savedSigns.has(m.label)
    save.setAttribute('aria-pressed', String(on))
    save.querySelector('span')!.textContent = on ? 'Saved' : 'Save'
  }
  sync()
  save.addEventListener('click', () => savedSigns.toggle(m.label))
  cardSubscriptions.push(savedSigns.onChange(sync))

  // example clip from dictionary.json, loaded only when asked for
  const exBtn = el.querySelector<HTMLButtonElement>('[data-example]')
  const exBox = el.querySelector<HTMLElement>('.example')!
  exBtn?.addEventListener('click', () => {
    const open = exBox.hidden
    exBox.hidden = !open
    exBtn.setAttribute('aria-expanded', String(open))
    if (open && !exBox.firstChild) {
      const v = document.createElement('video')
      v.controls = true
      v.playsInline = true
      v.loop = true
      v.muted = true
      v.src = e.clip!
      v.setAttribute('aria-label', `Example of the sign ${e.display}`)
      if (e.clip_captions) {
        const t = document.createElement('track')
        Object.assign(t, { kind: 'captions', src: e.clip_captions, srclang: 'en', label: 'English', default: true })
        v.append(t)
      }
      exBox.append(v)
      void v.play().catch(() => {})
    } else if (!open) exBox.querySelector('video')?.pause()
  })
  return el
}

let lastSource: ResultView['source'] = 'webcam'
export const resultSource = () => lastSource

export function renderResults(view: ResultView) {
  lastSource = view.source
  const unsure = view.matches[0].confidence < LOW_CONFIDENCE
  $('results-count').textContent = `${view.matches.length} possible matches`
  const secs = view.seconds.toFixed(1)
  $('clip-caption').textContent = `Your ${view.source === 'webcam' ? 'recorded' : 'selected'} ${secs} second clip plays on a loop`
  const cards = $('cards')
  cardSubscriptions.forEach((off) => off())
  cardSubscriptions = []
  cards.innerHTML = ''
  if (unsure) {
    const b = document.createElement('div')
    b.className = 'banner banner-warning'
    b.setAttribute('role', 'status')
    b.innerHTML = `${icon('info', 'icon')}<span>${
      view.source === 'webcam'
        ? 'Not sure. Try again with just the one sign inside the 2 seconds.'
        : 'Not sure. Try trimming tighter around the sign.'
    }</span>`
    cards.append(b)
  }
  view.matches.forEach((m, i) => cards.append(card(m, i + 1, unsure)))
  playLoop(view)
}
