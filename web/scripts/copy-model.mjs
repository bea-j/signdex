// Serves the model in use from public/ (so the app works offline):
//   model/model.onnx, model/labels.json, model/similar.json  ->  public/model/
// Missing model files only warn: the app shows "no model" instead of failing the build.
// Runs before `npm run dev` / `npm run build`. public/model/ is gitignored.
// (The onnxruntime-web .wasm is imported as a Vite asset in src/model.ts, not copied here.)
import { copyFile, mkdir, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const web = join(dirname(fileURLToPath(import.meta.url)), '..')
const modelDir = join(web, '..', 'model')
const exists = (p) => stat(p).then(() => true, () => false)

await mkdir(join(web, 'public', 'model'), { recursive: true })
for (const f of ['model.onnx', 'labels.json', 'similar.json']) {
  const src = join(modelDir, f)
  if (await exists(src)) await copyFile(src, join(web, 'public', 'model', f))
  else console.warn(`copy-model: ${src} not found (train a model first); the app will run without it`)
}
console.log('model ready in public/model')
