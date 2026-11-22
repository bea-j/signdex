// "My signs" screen and the header count.
import { entry, loadReferenceData } from './engine'
import { onEnter } from './router'
import { savedSigns } from './saved'

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const list = $('saved-list')
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`)

function updateCount() {
  const n = savedSigns.list().length
  const count = $('saved-count')
  count.textContent = String(n)
  count.setAttribute('aria-label', `${n} saved`)
}

function render() {
  const items = savedSigns.list()
  $('saved-empty').hidden = items.length > 0
  list.hidden = items.length === 0
  list.innerHTML = ''
  for (const label of items) {
    const e = entry(label)
    const li = document.createElement('li')
    li.className = 'saved-item glass-panel'
    li.innerHTML = `
      <div>
        <span class="sign-name">${esc(e.display!)}</span>
        ${e.meaning ? `<p class="meaning">${esc(e.meaning)}</p>` : ''}
      </div>
      <div class="card-actions">
        <a class="link" href="${esc(e.link!)}" target="_blank" rel="noreferrer" aria-label="See ${esc(e.display!)} in a dictionary (opens in a new tab)">See sign <svg class="icon-sm" aria-hidden="true"><use href="#i-external"/></svg></a>
        <button class="link" type="button" aria-label="Remove ${esc(e.display!)} from My signs">Remove</button>
      </div>`
    li.querySelector('button')!.addEventListener('click', () => savedSigns.toggle(label))
    list.append(li)
  }
}

savedSigns.onChange(() => {
  updateCount()
  if (!$('screen-saved').hidden) render()
})
updateCount()
onEnter('saved', () => {
  render()
  // display names and meanings come from dictionary.json
  void loadReferenceData().then(() => !$('screen-saved').hidden && render())
})
