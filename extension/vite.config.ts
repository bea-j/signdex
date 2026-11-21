import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url))

export default defineConfig({
  resolve: {
    // Shared landmark/feature/model code lives in the web app.
    alias: { '@shared': here('../web/src') },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        sidepanel: here('sidepanel.html'),
        offscreen: here('offscreen.html'),
        'capture-frame': here('capture-frame.html'),
        background: here('src/background.ts'),
        // Injected with chrome.scripting.executeScript({ files: [...] }): must stay classic scripts with no
        // imports (they only import types).
        probe: here('src/probe.ts'),
        'capture-video': here('src/capture-video.ts'),
      },
      output: {
        // MV3 manifest refers to background.js by a fixed name.
        entryFileNames: '[name].js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
})
