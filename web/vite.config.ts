import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    // src/spec.ts imports ../model/feature_spec.json, which lives outside web/.
    fs: { allow: ['..'] },
  },
})
