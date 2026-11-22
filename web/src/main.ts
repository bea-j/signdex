// Signdex main app. Screens: home, webcam and uploaded clip (each with its results panel), saved ("My signs").
import './app.css'
import './app/diag'
import './app/background'
import { currentScreen, show, type Screen } from './app/router'
import './app/demo'
import './app/dictionary'
import './app/my-signs'
import './app/quiz'
import './app/trim'
import { record } from './app/webcam'

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T

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

// Space or R: record (again) on the webcam screen, look up the selected window on the clip screen.
// Form fields keep their own keys.
const isShortcut = (e: KeyboardEvent) =>
  !e.repeat && !e.ctrlKey && !e.altKey && !e.metaKey && (e.code === 'Space' || e.key === ' ' || e.key.toLowerCase() === 'r')
const inField = (e: Event) =>
  !!(e.target as HTMLElement).closest('input, select, textarea, [contenteditable=""], [contenteditable="true"]')
const shortcutScreens: Screen[] = ['webcam', 'trim']

document.addEventListener('keydown', (e) => {
  if (!isShortcut(e) || inField(e) || !shortcutScreens.includes(currentScreen())) return
  e.preventDefault() // no scrolling and no click on whatever button has focus
  const s = currentScreen()
  if (s === 'webcam') void record()
  else $<HTMLButtonElement>('trim-lookup').click()
})
document.addEventListener('keyup', (e) => {
  if ((e.code === 'Space' || e.key === ' ') && !inField(e) && shortcutScreens.includes(currentScreen())) e.preventDefault()
})

// how-it-works steps: each card's width in proportion to the amount of text it holds
for (const card of document.querySelectorAll<HTMLElement>('.feature')) {
  card.style.setProperty('--grow', String(card.textContent?.replace(/\s+/g, ' ').trim().length ?? 1))
}

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
