import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url))

export default defineConfig({
  server: {
    // src/spec.ts imports ../model/feature_spec.json, which lives outside web/.
    fs: { allow: ['..'] },
  },
  build: {
    rollupOptions: {
      input: { main: here('index.html'), debug: here('debug.html') },
    },
  },
})
