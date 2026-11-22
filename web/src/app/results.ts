// Result cards (top 5) for the web app's results panels and the extension side panel.
import { LOW_CONFIDENCE, type SignMatch } from '../model'
import { entry, signerLink, similarTo } from './engine'
import { savedSigns } from './saved'
import { referenceClip, referenceWithheld } from '../reference'
import { prefersReducedMotion, SkeletonPlayer } from '../ui/skeleton'

const HIGH = 0.6

export function level(p: number): { cls: string; label: string } {
  if (p >= HIGH) return { cls: 'conf-high', label: 'High confidence' }
  if (p >= LOW_CONFIDENCE) return { cls: 'conf-amber', label: 'Medium confidence' }
  return { cls: 'conf-low', label: 'Low confidence' }
}

const icon = (name: string, cls = 'icon-sm') => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`)

// ---------------------------------------------------------------- cards
/** In place of an animation we won't show (two-handed sign, both hands never tracked): a labeled link tile. */
export function withheldTile(link: { href: string; aria: string }): HTMLAnchorElement {
  const a = document.createElement('a')
  a.className = 'ref ref-note'
  a.href = link.href
  a.target = '_blank'
  a.rel = 'noreferrer'
  a.title = 'No animation: our recordings of this two-handed sign don’t track both hands, so it would look wrong.'
  a.setAttribute('aria-label', `Two-handed sign, no animation. ${link.aria}`)
  a.innerHTML = `<span class="ref-note-title">Two-handed sign</span><span class="ref-note-link">Watch a real signer ${icon('external')}</span>`
  return a
}

/** Saved-state subscriptions of the cards in each container, dropped when it is re-rendered. */
const subscriptions = new WeakMap<HTMLElement, (() => void)[]>()

function card(m: SignMatch, rank: number, unsure: boolean, subs: (() => void)[]): HTMLElement {
  const e = entry(m.label)
  const lv = level(m.confidence)
  const pct = Math.round(m.confidence * 1000) / 10
  const pctText = pct >= 10 ? pct.toFixed(0) : pct.toFixed(1)
  const sims = similarTo(m.label)
  const link = signerLink(e)
  const el = document.createElement('article')
  el.className = `result-card glass-panel ${rank === 1 ? 'first' : ''}`
  el.setAttribute('aria-label', `Match ${rank}: ${e.display}, ${pctText} percent`)
  el.innerHTML = `
    <div class="card-row">
      <span class="rank" aria-hidden="true">${rank}</span>
      <button class="ref" type="button" hidden aria-pressed="false"
        aria-label="Play a reference signer's movement for ${esc(e.display!)}"
        title="Reference: a PopSign ASL signer (CC BY 4.0), as landmarks"><canvas aria-hidden="true"></canvas></button>
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
            <a class="link" href="${esc(link.href)}" target="_blank" rel="noreferrer" aria-label="${esc(link.aria)}">Watch a real signer ${icon('external')}</a>
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
  subs.push(savedSigns.onChange(sync))

  // reference animation (skeleton of a PopSign training signer): the top card loops, the others play on hover/tap
  const refBtn = el.querySelector<HTMLButtonElement>('.ref')!
  void referenceClip(m.label).then(async (clip) => {
    if (!el.isConnected) return
    if (!clip) {
      if (await referenceWithheld(m.label)) refBtn.replaceWith(withheldTile(signerLink(e)))
      return
    }
    refBtn.hidden = false
    const player = new SkeletonPlayer(refBtn.querySelector('canvas')!, clip)
    subs.push(() => player.pause())
    let pinned = rank === 1 && !prefersReducedMotion()
    const sync = () => refBtn.setAttribute('aria-pressed', String(player.isPlaying))
    if (pinned) player.play()
    else player.still()
    sync()
    el.addEventListener('pointerenter', (ev) => {
      if (ev.pointerType === 'mouse' && !player.isPlaying) player.play(0)
      sync()
    })
    el.addEventListener('pointerleave', (ev) => {
      if (ev.pointerType === 'mouse' && !pinned) player.still()
      sync()
    })
    refBtn.addEventListener('click', () => {
      pinned = !pinned
      if (pinned) player.play(0)
      else player.still()
      sync()
    })
  })

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

/**
 * Top-k result cards into `container` (web app results screen and the extension side panel).
 * Below LOW_CONFIDENCE on the top match, a warning banner with `unsureText` comes first.
 */
export function renderCards(container: HTMLElement, matches: SignMatch[], unsureText: string) {
  const unsure = matches[0].confidence < LOW_CONFIDENCE
  subscriptions.get(container)?.forEach((off) => off())
  const subs: (() => void)[] = []
  subscriptions.set(container, subs)
  container.innerHTML = ''
  if (unsure) {
    const b = document.createElement('div')
    b.className = 'banner banner-warning'
    b.setAttribute('role', 'status')
    b.innerHTML = `${icon('info', 'icon')}<span>${esc(unsureText)}</span>`
    container.append(b)
  }
  matches.forEach((m, i) => container.append(card(m, i + 1, unsure, subs)))
}
