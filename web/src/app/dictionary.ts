// Dictionary page: all 250 signs in a searchable grid, each with its looping reference animation, meaning and
// "Watch a real signer" link (dictionary.json + the reference animations). Only cards on screen animate.
import { preloadAllReferences, referenceClip } from '../reference'
import { prefersReducedMotion, SkeletonPlayer } from '../ui/skeleton'
import { entry, knownLabels, loadReferenceData, signerLink } from './engine'
import { onEnter, onLeave } from './router'

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const grid = $('dict-grid')
const search = $<HTMLInputElement>('dict-search')
const pause = $<HTMLInputElement>('dict-pause')
const PAUSE_KEY = 'signdex.dictionaryPaused'

interface Card {
  label: string
  name: string
  el: HTMLElement
  canvas: HTMLCanvasElement
  player?: SkeletonPlayer
  visible: boolean
}
let cards: Card[] = []
let built = false
let active = false

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`)

try {
  pause.checked = localStorage.getItem(PAUSE_KEY) === '1' || prefersReducedMotion()
} catch {
  pause.checked = prefersReducedMotion()
}
const animating = () => active && !pause.checked

/** Start or stop a card's animation to match whether it is on screen and animations are on. */
function sync(card: Card) {
  if (!card.player) return
  if (card.visible && animating()) {
    if (!card.player.isPlaying) card.player.play()
  } else if (card.player.isPlaying || !card.visible) {
    card.player.still()
  }
}

const observer = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      const card = cards.find((c) => c.el === e.target)
      if (!card) continue
      card.visible = e.isIntersecting
      if (card.visible && !card.player) {
        void referenceClip(card.label).then((clip) => {
          if (!clip) {
            card.canvas.parentElement!.hidden = true
            return
          }
          card.player = new SkeletonPlayer(card.canvas, clip)
          card.player.still()
          sync(card)
        })
      } else sync(card)
    }
  },
  { rootMargin: '200px 0px' },
)

async function build() {
  built = true
  void preloadAllReferences() // one ~1 MB request instead of one per sign; clip loads wait for it
  await loadReferenceData()
  const labels = knownLabels()
  cards = labels
    .map((label) => ({ label, e: entry(label) }))
    .sort((a, b) => a.e.display!.localeCompare(b.e.display!))
    .map(({ label, e }) => {
      const link = signerLink(e)
      const li = document.createElement('li')
      li.className = 'dict-card glass-panel'
      li.innerHTML = `
        <div class="ref dict-ref" aria-hidden="true"><canvas></canvas></div>
        <div class="dict-text">
          <h2 class="sign-name">${esc(e.display!)}</h2>
          ${e.meaning ? `<p class="meaning">${esc(e.meaning)}</p>` : ''}
          <a class="link" href="${esc(link.href)}" target="_blank" rel="noreferrer" aria-label="${esc(link.aria)}">Watch a real signer <svg class="icon-sm" aria-hidden="true"><use href="#i-external"/></svg></a>
        </div>`
      grid.append(li)
      observer.observe(li)
      return { label, name: `${e.display} ${label}`.toLowerCase(), el: li, canvas: li.querySelector('canvas')!, visible: false }
    })
  filter()
}

function filter() {
  const q = search.value.trim().toLowerCase()
  let shown = 0
  for (const c of cards) {
    const match = !q || c.name.includes(q)
    c.el.hidden = !match
    if (match) shown++
  }
  $('dict-count').textContent = q ? `${shown} of ${cards.length} signs match “${search.value.trim()}”` : `${cards.length} signs`
  $('dict-empty').hidden = shown > 0
}

search.addEventListener('input', filter)
pause.addEventListener('change', () => {
  try {
    localStorage.setItem(PAUSE_KEY, pause.checked ? '1' : '0')
  } catch {
    /* this visit only */
  }
  cards.forEach(sync)
})

onEnter('dictionary', () => {
  active = true
  if (!built) void build()
  else cards.forEach(sync)
})
onLeave('dictionary', () => {
  active = false
  cards.forEach(sync)
})
