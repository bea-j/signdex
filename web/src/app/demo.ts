// Home page demo in the hero panel: a reference skeleton signs a word, its hand points light up, then the result
// card fills in with the name and the model's confidence; cycles through a few reliable signs. Uses the same
// reference animation data as the result cards (only these signs' bytes are fetched). With
// prefers-reduced-motion it shows one still frame with the card filled. Without reference data the static
// illustration stays.
import { referenceClip, referenceSigns, type RefClip } from '../reference'
import { prefersReducedMotion, SkeletonPlayer } from '../ui/skeleton'
import { entry, loadReferenceData } from './engine'
import { onEnter, onLeave } from './router'

const SIGNS = ['hat', 'cat', 'drink', 'yes']
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const stage = $('demo-stage')
const canvas = $<HTMLCanvasElement>('demo-canvas')
const name = $('demo-name')
const pill = $('demo-pill')
const bar = $('demo-bar')
const barWrap = $('demo-barwrap')
const badgeText = $('demo-badge-text')

let run = 0 // bumped to stop a running cycle
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

function card(clip: RefClip | null) {
  if (!clip) {
    name.textContent = '…'
    name.classList.add('pending')
    pill.textContent = 'matching…'
    bar.style.width = '0%'
    return
  }
  const pct = Math.round(clip.confidence * 100)
  name.textContent = entry(clip.label).display
  name.classList.remove('pending')
  pill.textContent = `${pct}% match`
  barWrap.className = `bar ${clip.confidence >= 0.6 ? 'conf-high' : 'conf-amber'}`
  bar.style.width = `${pct}%`
}

/** Play the clip once (resolves at its last frame), with the hands lighting up as it goes. */
function playOnce(player: SkeletonPlayer, my: number): Promise<void> {
  return new Promise((resolve) => {
    player.onFrame = (f, n) => {
      player.highlight = Math.min(1, f / Math.max(1, n * 0.35))
      if (my !== run) {
        player.pause()
        resolve()
      } else if (f >= n - 1) {
        player.pause()
        resolve()
      }
    }
    player.play(0)
  })
}

async function cycle() {
  const my = ++run
  await loadReferenceData() // display names from dictionary.json
  // the chosen signs, topped up with the most confident others if some are missing from the reference file
  const available = await referenceSigns()
  const have = new Set(available.map((s) => s.label))
  const labels = SIGNS.filter((s) => have.has(s))
  for (const s of [...available].sort((a, b) => b.confidence - a.confidence)) {
    if (labels.length >= SIGNS.length) break
    if (!labels.includes(s.label)) labels.push(s.label)
  }
  const clips = (await Promise.all(labels.map(referenceClip))).filter((c): c is RefClip => !!c)
  if (!clips.length || my !== run) return
  stage.classList.add('has-demo')
  if (prefersReducedMotion()) {
    const p = new SkeletonPlayer(canvas, clips[0], { holdMs: 0 })
    p.highlight = 1
    p.still()
    badgeText.textContent = 'Hands detected'
    card(clips[0])
    return
  }
  for (let i = 0; my === run; i = (i + 1) % clips.length) {
    const clip = clips[i]
    const player = new SkeletonPlayer(canvas, clip, { holdMs: 0 })
    card(null)
    badgeText.textContent = 'Tracking hands and body'
    player.highlight = 0
    player.draw(0)
    await sleep(500)
    if (my !== run) return
    await playOnce(player, my)
    if (my !== run) return
    badgeText.textContent = 'Hands detected'
    card(clip)
    await sleep(2200)
  }
}

function stop() {
  run++
}

onEnter('home', () => void cycle())
onLeave('home', stop)
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stop()
  else if (!$('screen-home').hidden) void cycle()
})
new ResizeObserver(() => stage.classList.contains('has-demo') && void cycle()).observe(stage)
