// Signdex main app. Screens: home, webcam, trim (uploaded clip), results, saved ("My signs").
import './app.css'
import { resultSource, stopResults } from './app/results'
import { currentScreen, onLeave, show, type Screen } from './app/router'
import './app/my-signs'
import './app/trim'
import { record, recordAgain } from './app/webcam'

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

// results: "Try another sign" goes back to where the clip came from
function tryAgain() {
  if (resultSource() === 'webcam') recordAgain()
  else show('trim')
}
$('try-another').addEventListener('click', tryAgain)
onLeave('results', stopResults)

// Space or R: record (webcam), look up (trim), try again (results). Form fields keep their own keys.
const isShortcut = (e: KeyboardEvent) =>
  !e.repeat && !e.ctrlKey && !e.altKey && !e.metaKey && (e.code === 'Space' || e.key === ' ' || e.key.toLowerCase() === 'r')
const inField = (e: Event) =>
  !!(e.target as HTMLElement).closest('input, select, textarea, [contenteditable=""], [contenteditable="true"]')
const shortcutScreens: Screen[] = ['webcam', 'trim', 'results']

document.addEventListener('keydown', (e) => {
  if (!isShortcut(e) || inField(e) || !shortcutScreens.includes(currentScreen())) return
  e.preventDefault() // no scrolling and no click on whatever button has focus
  const s = currentScreen()
  if (s === 'webcam') void record()
  else if (s === 'trim') $<HTMLButtonElement>('trim-lookup').click()
  else tryAgain()
})
document.addEventListener('keyup', (e) => {
  if ((e.code === 'Space' || e.key === ' ') && !inField(e) && shortcutScreens.includes(currentScreen())) e.preventDefault()
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
