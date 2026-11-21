// Signdex main app: screens (home, webcam, trim, results, saved), switched by showing one <section> at a time.
import './app.css'

export type Screen = 'home' | 'webcam' | 'trim' | 'results' | 'saved'

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const screens: Screen[] = ['home', 'webcam', 'trim', 'results', 'saved']
let current: Screen = 'home'
const leaveHandlers: Partial<Record<Screen, () => void>> = {}
const enterHandlers: Partial<Record<Screen, () => void>> = {}

export function onEnter(s: Screen, f: () => void) {
  enterHandlers[s] = f
}
export function onLeave(s: Screen, f: () => void) {
  leaveHandlers[s] = f
}
export function currentScreen(): Screen {
  return current
}

export function show(s: Screen, focusTitle = true) {
  if (s !== current) leaveHandlers[current]?.()
  current = s
  for (const id of screens) $(`screen-${id}`).hidden = id !== s
  if (s === 'saved') $('nav-saved').setAttribute('aria-current', 'page')
  else $('nav-saved').removeAttribute('aria-current')
  window.scrollTo({ top: 0 })
  enterHandlers[s]?.()
  if (focusTitle && s !== 'home') {
    // move focus to the new screen's heading so keyboard and screen reader users land in the right place
    const h = document.querySelector<HTMLElement>(`#screen-${s} h1`)
    if (h) {
      h.tabIndex = -1
      h.focus({ preventScroll: true })
    }
  }
}

// data-go="home|webcam|saved|how" buttons and links
document.addEventListener('click', (e) => {
  const el = (e.target as HTMLElement).closest<HTMLElement>('[data-go]')
  if (!el) return
  e.preventDefault()
  const go = el.dataset.go!
  if (go === 'how') {
    show('home', false)
    $('how').scrollIntoView({ behavior: 'smooth' })
  } else show(go as Screen)
})

// feature cards: pointer glow + slight tilt, as in the design
for (const card of document.querySelectorAll<HTMLElement>('.feature')) {
  card.addEventListener('pointermove', (e) => {
    const b = card.getBoundingClientRect()
    const x = e.clientX - b.left
    const y = e.clientY - b.top
    card.style.setProperty('--mouse-x', `${x}px`)
    card.style.setProperty('--mouse-y', `${y}px`)
    card.style.setProperty('--tilt-x', `${(y / b.height - 0.5) * -2.5}deg`)
    card.style.setProperty('--tilt-y', `${(x / b.width - 0.5) * 2.5}deg`)
  })
  card.addEventListener('pointerleave', () => {
    card.style.setProperty('--tilt-x', '0deg')
    card.style.setProperty('--tilt-y', '0deg')
  })
}

show('home', false)
