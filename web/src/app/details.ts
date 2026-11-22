// "Show details" stats panel under the camera/player: live capture stats and the last lookup, as design tiles.
import type { LookupResult } from '../lookup'
import { COORDS_PER_POINT as C, OFFSETS, TRAIN_H_OVER_W, type KaggleFrame } from '../landmarks'

export interface Stat {
  label: string
  value: string
  /** Optional colored dot: left/right hand slot colors, or detected/missing. */
  tone?: 'left' | 'right' | 'ok' | 'off'
}
export interface StatGroup {
  title: string
  stats: Stat[]
}

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`)

export function renderDetails(el: HTMLElement, groups: StatGroup[]) {
  el.innerHTML = groups
    .filter((g) => g.stats.length)
    .map(
      (g) => `<div class="stat-group"><p class="stat-title">${esc(g.title)}</p><dl class="stats">${g.stats
        .map(
          (s) =>
            `<div class="stat"><dt>${esc(s.label)}</dt><dd>${s.tone ? `<span class="tone tone-${s.tone}" aria-hidden="true"></span>` : ''}${esc(s.value)}</dd></div>`,
        )
        .join('')}</dl></div>`,
    )
    .join('')
}

const present = (f: KaggleFrame, offset: number, n: number) => {
  for (let i = 0; i < n; i++) if (!Number.isNaN(f[(offset + i) * C])) return true
  return false
}
export const parts = (f: KaggleFrame) => ({
  left: present(f, OFFSETS.left_hand, 21),
  right: present(f, OFFSETS.right_hand, 21),
  lips: present(f, OFFSETS.face, 468),
  pose: present(f, OFFSETS.pose, 33),
})

/** Frame size and the y scaling that maps it to the training video's aspect (toTrainingAspect). */
export function frameStats(width: number, height: number): Stat[] {
  if (!width || !height) return []
  return [
    { label: 'Frame size', value: `${width} × ${height} (H/W ${(height / width).toFixed(2)})` },
    { label: 'Aspect scaling', value: `y × ${(height / width / TRAIN_H_OVER_W).toFixed(3)} to training H/W ${TRAIN_H_OVER_W}` },
  ]
}

/** Detection of each part: live (one frame) or as a count over a clip. */
export function partStats(frames: KaggleFrame[], live: boolean): Stat[] {
  if (!frames.length) return []
  const counts = { left: 0, right: 0, lips: 0, pose: 0 }
  for (const f of frames) {
    const p = parts(f)
    for (const k of Object.keys(counts) as (keyof typeof counts)[]) if (p[k]) counts[k]++
  }
  const v = (n: number) => (live ? (n ? 'detected' : 'missing') : `${n} of ${frames.length} frames`)
  return [
    { label: 'Left hand slot', value: v(counts.left), tone: counts.left ? 'left' : 'off' },
    { label: 'Right hand slot', value: v(counts.right), tone: counts.right ? 'right' : 'off' },
    { label: 'Lips', value: v(counts.lips), tone: counts.lips ? 'ok' : 'off' },
    { label: 'Pose', value: v(counts.pose), tone: counts.pose ? 'ok' : 'off' },
  ]
}

export function lookupStats(recorded: number, res: LookupResult, modelLength: number): Stat[] {
  const t = res.timing
  return [
    { label: 'Frames', value: `${recorded} recorded → ${res.stage1.T} after trim` },
    { label: 'Frames with hands', value: `${res.handFrames} of ${recorded}` },
    {
      label: 'Dominant hand',
      value: res.stage1.mirrored ? 'left (mirrored to right)' : 'right (not mirrored)',
      tone: res.stage1.mirrored ? 'left' : 'right',
    },
    { label: 'Lookup time', value: `${Math.round(t.totalMs)} ms (features ${Math.round(t.featuresMs)} + model ${Math.round(t.modelMs)})` },
    { label: 'Test-time averaging', value: res.variants === 1 ? 'off (clip too short)' : `${res.variants} trimmed versions` },
    { label: 'Model frame length', value: `${modelLength} frames × 222 features` },
  ]
}
