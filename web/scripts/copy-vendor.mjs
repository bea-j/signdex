// Self-hosts third-party browser scripts that are loaded as plain <script> tags (not bundled):
//   p5 1.x (from npm) -> public/vendor/p5.min.js, used by the animated background (public/bg/signdex-bg.js).
// Runs before `npm run dev` / `npm run build`. public/vendor/ is gitignored.
import { copyFile, mkdir } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const web = join(dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(import.meta.url)
const p5Dir = dirname(require.resolve('p5/package.json'))
await mkdir(join(web, 'public', 'vendor'), { recursive: true })
await copyFile(join(p5Dir, 'lib', 'p5.min.js'), join(web, 'public', 'vendor', 'p5.min.js'))
console.log(`p5 ${require('p5/package.json').version} ready in public/vendor`)
