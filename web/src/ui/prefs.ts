// Display preferences (off by default, remembered in localStorage): overlay layers and the details panel.
// Every <input type="checkbox" data-pref="…"> on the page is kept in sync with its preference.
export type PrefName = 'points' | 'face' | 'details'
const KEY: Record<PrefName, string> = {
  points: 'signdex.showPoints',
  face: 'signdex.showFace',
  details: 'signdex.showDetails',
}
const listeners = new Set<() => void>()

function read(name: PrefName): boolean {
  try {
    return localStorage.getItem(KEY[name]) === '1'
  } catch {
    return false
  }
}
const values: Record<PrefName, boolean> = { points: read('points'), face: read('face'), details: read('details') }

export const prefs = {
  get: (name: PrefName) => values[name],
  set(name: PrefName, on: boolean) {
    values[name] = on
    try {
      localStorage.setItem(KEY[name], on ? '1' : '0')
    } catch {
      /* keep for this visit only */
    }
    sync()
    listeners.forEach((f) => f())
  },
  onChange(f: () => void) {
    listeners.add(f)
  },
}

function sync() {
  for (const el of document.querySelectorAll<HTMLInputElement>('input[data-pref]')) {
    el.checked = values[el.dataset.pref as PrefName]
  }
}

document.addEventListener('change', (e) => {
  const el = e.target as HTMLInputElement
  if (el.matches?.('input[data-pref]')) prefs.set(el.dataset.pref as PrefName, el.checked)
})
sync()
