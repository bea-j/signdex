// Content script (injected on the hotkey into every frame): can the page's <video> be read through a canvas?
// Draws the largest <video> to a canvas and checks whether the canvas is tainted or blank.
// The IIFE's return value is the script's completion value, which chrome.scripting.executeScript returns
// per frame to the service worker. Wrapped in an IIFE so re-injecting doesn't redeclare anything.
import type { ProbeResult } from './messages'
;(() => {
  const isTop = window === window.top
  const videos = Array.from(document.querySelectorAll('video'))
  if (videos.length === 0 && !isTop) return undefined // stay quiet in frames without video

  const result: ProbeResult = {
    frameUrl: location.origin + location.pathname,
    isTopFrame: isTop,
    videos: videos.length,
    viewport: { width: window.innerWidth, height: window.innerHeight, dpr: window.devicePixelRatio },
    tainted: null,
  }
  const v = videos
    .filter((x) => x.videoWidth > 0)
    .sort((a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight)[0]

  if (v) {
    const r = v.getBoundingClientRect()
    result.rect = { x: r.left, y: r.top, width: r.width, height: r.height }
    result.chosen = {
      width: v.videoWidth,
      height: v.videoHeight,
      src: (v.currentSrc || '').slice(0, 80),
      paused: v.paused,
      readyState: v.readyState,
    }
    const c = document.createElement('canvas')
    c.width = 160
    c.height = Math.max(1, Math.round((160 * v.videoHeight) / v.videoWidth))
    const ctx = c.getContext('2d')!
    try {
      ctx.drawImage(v, 0, 0, c.width, c.height)
      const px = ctx.getImageData(0, 0, c.width, c.height).data // throws SecurityError if tainted
      result.tainted = false
      let sum = 0
      for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2]
      result.blank = sum / (px.length / 4) / 3 < 4
      result.thumb = c.toDataURL('image/jpeg', 0.6)
    } catch (e) {
      const err = e as DOMException
      result.tainted = err.name === 'SecurityError' ? true : null
      result.error = `${err.name}: ${err.message}`
    }
  }

  console.log('[Signdex probe]', result.tainted === true ? 'canvas TAINTED' : result.tainted === false ? (result.blank ? 'canvas readable but BLANK' : 'canvas readable') : 'no drawable video', result)
  return result
})()
