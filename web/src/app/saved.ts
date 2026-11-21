// "My signs": saved sign labels, kept in localStorage on this device only.
// Storage can be unavailable (private mode, blocked site data): the list then just lasts for this visit.
const KEY = 'signdex.saved'
let items: string[] = read()
const listeners = new Set<() => void>()

function read(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? '[]')
    return Array.isArray(v) ? v.filter((x) => typeof x === 'string') : []
  } catch {
    return []
  }
}

function write() {
  try {
    localStorage.setItem(KEY, JSON.stringify(items))
  } catch {
    /* keep the in-memory list */
  }
  listeners.forEach((f) => f())
}

export const savedSigns = {
  list: (): string[] => [...items],
  has: (label: string) => items.includes(label),
  toggle(label: string) {
    items = items.includes(label) ? items.filter((x) => x !== label) : [...items, label]
    write()
  },
  /** Returns an unsubscribe function. */
  onChange(f: () => void): () => void {
    listeners.add(f)
    return () => listeners.delete(f)
  },
}
