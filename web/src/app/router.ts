// Screens (home, webcam, trim, saved, quiz): one <section id="screen-…"> visible at a time.
export type Screen = 'home' | 'webcam' | 'trim' | 'saved' | 'quiz' | 'guide' | 'privacy'

const screens: Screen[] = ['home', 'webcam', 'trim', 'saved', 'quiz', 'guide', 'privacy']
let current: Screen = 'home'
// several modules may hook the same screen, so keep a list of handlers per screen
const enter: Partial<Record<Screen, (() => void)[]>> = {}
const leave: Partial<Record<Screen, (() => void)[]>> = {}

export const onEnter = (s: Screen, f: () => void) => void (enter[s] ??= []).push(f)
export const onLeave = (s: Screen, f: () => void) => void (leave[s] ??= []).push(f)
export const currentScreen = () => current

export function show(s: Screen, focusTitle = true) {
  if (s !== current) leave[current]?.forEach((f) => f())
  current = s
  for (const id of screens) document.getElementById(`screen-${id}`)!.hidden = id !== s
  const nav = document.getElementById('nav-saved')!
  if (s === 'saved') nav.setAttribute('aria-current', 'page')
  else nav.removeAttribute('aria-current')
  window.scrollTo({ top: 0 })
  enter[s]?.forEach((f) => f())
  if (focusTitle && s !== 'home') {
    // land keyboard and screen reader users on the new screen's heading
    const h = document.querySelector<HTMLElement>(`#screen-${s} h1`)
    if (h) {
      h.tabIndex = -1
      h.focus({ preventScroll: true })
    }
  }
}
