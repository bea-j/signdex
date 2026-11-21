// Copies the MediaPipe wasm runtime and downloads the three landmarker models into public/mediapipe/,
// so the app serves everything itself (works offline at the demo). Skips files that already exist.
// Runs automatically before `npm run dev` / `npm run build`.
import { createRequire } from 'node:module'
import { cp, mkdir, stat, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const web = join(dirname(fileURLToPath(import.meta.url)), '..')
const dest = join(web, 'public', 'mediapipe')
const pkgDir = dirname(require.resolve('@mediapipe/tasks-vision'))

// Pinned model versions ("1"); "latest" only as a fallback if a pinned URL disappears.
const G = 'https://storage.googleapis.com/mediapipe-models'
const MODELS = {
  'hand_landmarker.task': `${G}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
  'pose_landmarker_full.task': `${G}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
  'pose_landmarker_lite.task': `${G}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  'face_landmarker.task': `${G}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
}

const exists = (p) => stat(p).then((s) => s.size > 0, () => false)

await mkdir(dest, { recursive: true })
await cp(join(pkgDir, 'wasm'), join(dest, 'wasm'), { recursive: true })

for (const [name, url] of Object.entries(MODELS)) {
  const out = join(dest, name)
  if (await exists(out)) continue
  let res = await fetch(url)
  if (!res.ok) res = await fetch(url.replace('/float16/1/', '/float16/latest/'))
  if (!res.ok) throw new Error(`download failed for ${name}: HTTP ${res.status}`)
  await writeFile(out, Buffer.from(await res.arrayBuffer()))
  console.log(`fetched ${name}`)
}
console.log('mediapipe assets ready in public/mediapipe')
