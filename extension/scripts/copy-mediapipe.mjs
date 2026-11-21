// Copies the MediaPipe wasm runtime and the three landmarker models from web/public/mediapipe (fetched by
// web/scripts/fetch-mediapipe.mjs) into extension/public/mediapipe, so the extension bundles them.
import { cp, mkdir, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ext = join(dirname(fileURLToPath(import.meta.url)), '..')
const src = join(ext, '..', 'web', 'public', 'mediapipe')
const dest = join(ext, 'public', 'mediapipe')
const FILES = ['wasm', 'hand_landmarker.task', 'pose_landmarker_full.task', 'face_landmarker.task']

await mkdir(dest, { recursive: true })
for (const f of FILES) {
  await stat(join(src, f)) // throws with a clear path if the web fetch step did not run
  await cp(join(src, f), join(dest, f), { recursive: true })
}
console.log('mediapipe assets copied to extension/public/mediapipe')
