// demo/dictionary.csv -> web/public/dictionary.json (keyed by the model's label), checked against labels.json.
//   npm run dictionary            (from web/)
// Refuses to write if any label is missing, extra, duplicated or misspelled. Columns used:
//   label, display_name, meaning, dictionary_url, example_clip_url, practice_sign
// checked_by and notes are editorial and stay out of the public JSON.
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const repo = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
const csvPath = process.argv[2] ?? join(repo, 'demo', 'dictionary.csv')
const out = join(repo, 'web', 'public', 'dictionary.json')
const labelsPath = [join(repo, 'model', 'labels.json'), join(repo, 'web', 'public', 'model', 'labels.json')].find(existsSync)
if (!labelsPath) throw new Error('labels.json not found in model/ or web/public/model/')

/** RFC 4180 CSV: quoted fields, "" escapes, commas and newlines inside quotes. */
function parseCsv(text) {
  const rows = []
  let row = []
  let field = ''
  let quoted = false
  text = text.replace(/^﻿/, '')
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') (field += '"'), i++
      else if (c === '"') quoted = false
      else field += c
    } else if (c === '"') quoted = true
    else if (c === ',') row.push(field), (field = '')
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field), rows.push(row), (row = []), (field = '')
    } else field += c
  }
  if (field !== '' || row.length) row.push(field), rows.push(row)
  return rows.filter((r) => r.some((f) => f.trim() !== ''))
}

const [header, ...rows] = parseCsv(readFileSync(csvPath, 'utf8'))
const col = Object.fromEntries(header.map((h, i) => [h.trim(), i]))
for (const need of ['label', 'display_name', 'meaning', 'dictionary_url']) {
  if (!(need in col)) throw new Error(`${csvPath}: missing column "${need}" (have: ${header.join(', ')})`)
}
const get = (r, name) => (name in col ? (r[col[name]] ?? '').trim() : '')

const labels = JSON.parse(readFileSync(labelsPath, 'utf8'))
const known = new Set(labels)
const problems = []
const seen = new Map()
const signs = {}
const isUrl = (s) => /^https?:\/\/\S+$/i.test(s) || /^\/\S+$/.test(s)

rows.forEach((r, i) => {
  const line = i + 2
  const label = get(r, 'label')
  if (r.length !== header.length) problems.push(`line ${line}: ${r.length} fields, header has ${header.length} (unquoted comma?)`)
  if (!known.has(label)) {
    const near = labels.find((l) => l.toLowerCase() === label.toLowerCase())
    problems.push(`line ${line}: label "${label}" is not in labels.json${near ? ` (did you mean "${near}"?)` : ''}`)
    return
  }
  if (seen.has(label)) problems.push(`line ${line}: duplicate label "${label}" (first on line ${seen.get(label)})`)
  seen.set(label, line)
  const entry = {
    display: get(r, 'display_name') || null,
    meaning: get(r, 'meaning') || null,
    link: get(r, 'dictionary_url') || null,
    clip: get(r, 'example_clip_url') || null,
    clip_captions: null,
    practice: /^(y|yes|true|1)$/i.test(get(r, 'practice_sign')),
  }
  for (const [k, v] of [['dictionary_url', entry.link], ['example_clip_url', entry.clip]]) {
    if (v && !isUrl(v)) problems.push(`line ${line} (${label}): ${k} is not an http(s) URL or /path: "${v}"`)
  }
  if (!entry.display) problems.push(`line ${line} (${label}): empty display_name`)
  signs[label] = entry
})
for (const l of labels) if (!seen.has(l)) problems.push(`label "${l}" from labels.json has no row in the CSV`)

if (problems.length) {
  console.error(`${problems.length} problem(s); ${out} not written:\n  ` + problems.join('\n  '))
  process.exit(1)
}

const sorted = Object.fromEntries(Object.keys(signs).sort((a, b) => a.localeCompare(b)).map((k) => [k, signs[k]]))
const doc = {
  _readme: [
    'Generated from demo/dictionary.csv by web/scripts/build-dictionary.mjs (npm run dictionary); edit the CSV, not this file.',
    'Keyed by the model label. display/meaning are shown on result cards; link is the "Watch a real signer" link;',
    'clip is an optional example clip (URL or /path under web/public), clip_captions an optional WebVTT file;',
    'practice marks signs for the practice/quiz list.',
  ],
  signs: sorted,
}
writeFileSync(out, JSON.stringify(doc, null, 1) + '\n')
const filled = (k) => Object.values(sorted).filter((e) => e[k]).length
console.log(
  `wrote ${out}: ${labels.length}/${labels.length} labels match labels.json; ` +
    `display ${filled('display')}, meaning ${filled('meaning')}, link ${filled('link')}, clip ${filled('clip')}, practice ${filled('practice')}`,
)
