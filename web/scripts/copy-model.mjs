// Serves the model in use and the onnxruntime-web wasm runtime from public/ (so the app works offline):
//   model/model.onnx, model/labels.json, model/similar.json  ->  public/model/
//   onnxruntime-web wasm runtime                              ->  public/ort/
// Missing model files only warn: the app shows "no model" instead of failing the build.
// Runs before `npm run dev` / `npm run build`. public/model/ and public/ort/ are gitignored.
import { createRequire } from 'node:module'
import { copyFile, mkdir, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const web = join(dirname(fileURLToPath(import.meta.url)), '..')
const modelDir = join(web, '..', 'model')
const exists = (p) => stat(p).then(() => true, () => false)

await mkdir(join(web, 'public', 'model'), { recursive: true })
for (const f of ['model.onnx', 'labels.json', 'similar.json']) {
  const src = join(modelDir, f)
  if (await exists(src)) await copyFile(src, join(web, 'public', 'model', f))
  else console.warn(`copy-model: ${src} not found (train a model first); the app will run without it`)
}

const ortDist = dirname(require.resolve('onnxruntime-web/ort-wasm-simd-threaded.wasm'))
await mkdir(join(web, 'public', 'ort'), { recursive: true })
for (const f of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) {
  await copyFile(join(ortDist, f), join(web, 'public', 'ort', f))
}
console.log('model + onnxruntime-web runtime ready in public/model and public/ort')
