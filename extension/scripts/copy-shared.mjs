// Copies what the side panel's sign lookup needs from the web app into extension/public:
//   model/{model.onnx, labels.json, similar.json}  (web/public/model, filled by web/scripts/copy-model.mjs)
//   dictionary.json                                  (web/public, from demo/dictionary.csv)
//   brand/signdex-logo.png, ambient/ambient-main.png (the design's logo and background)
//   reference/signs.bin                              (reference animations, from model/reference.py)
// The copies are gitignored; the web app's files are the source of truth.
import { cp, mkdir, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ext = join(dirname(fileURLToPath(import.meta.url)), '..')
const web = join(ext, '..', 'web', 'public')
const dest = join(ext, 'public')
const FILES = [
  'model/model.onnx',
  'model/labels.json',
  'model/similar.json',
  'dictionary.json',
  'brand/signdex-logo.png',
  'brand/favicon.svg',
  'ambient/ambient-main.png',
  'reference/signs.bin', // reference animations in the result cards (model/reference.py)
]

let missing = 0
for (const f of FILES) {
  try {
    await stat(join(web, f))
  } catch {
    console.warn(`copy-shared: ${join(web, f)} is missing; the side panel lookup will not work without it`)
    missing++
    continue
  }
  await mkdir(dirname(join(dest, f)), { recursive: true })
  await cp(join(web, f), join(dest, f))
}
console.log(`copy-shared: ${FILES.length - missing}/${FILES.length} files copied to extension/public`)
