// Content script (injected into the frame that owns the video). Two jobs:
//  start-video: PRIMARY path. Draw the page's <video> to a canvas at native resolution (~20 fps) and transfer each
//               frame as an ImageBitmap to a hidden extension iframe that runs MediaPipe. Re-checks for taint/blank
//               every 2 s and reports failure so the service worker can switch to tab capture.
//  track-rect:  FALLBACK helper. Report the video's on-screen rect (as viewport fractions) for cropping tab capture.
// Only imports types, so the bundle stays a classic script for chrome.scripting.executeScript.
import type { Crop, CsMsg, PortMsg } from './messages'

declare global {
  // eslint-disable-next-line no-var
  var __signseekCapture: boolean | undefined
}

;(() => {
  if (globalThis.__signseekCapture) return // already injected into this frame's isolated world
  globalThis.__signseekCapture = true

  const FPS = 20
  const CHECK_EVERY_MS = 2000
  let stopFns: (() => void)[] = []
  const stopAll = () => {
    stopFns.forEach((f) => f())
    stopFns = []
  }

  const pickVideo = (): HTMLVideoElement | undefined =>
    Array.from(document.querySelectorAll('video'))
      .filter((v) => v.videoWidth > 0)
      .sort((a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight)[0]

  const failed = (reason: string) => {
    stopAll()
    chrome.runtime.sendMessage({ target: 'sw', type: 'video-path-failed', reason }).catch(() => {})
  }

  /** Returns null if readable, else why not. Uses a tiny canvas so the check is cheap. */
  function check(video: HTMLVideoElement): string | null {
    const c = document.createElement('canvas')
    c.width = 32
    c.height = 18
    const ctx = c.getContext('2d')!
    try {
      ctx.drawImage(video, 0, 0, 32, 18)
      const px = ctx.getImageData(0, 0, 32, 18).data
      let sum = 0
      for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2]
      return sum / (px.length / 4) / 3 < 4 ? 'blank' : null
    } catch (e) {
      return (e as DOMException).name === 'SecurityError' ? 'tainted' : `draw failed: ${(e as Error).message}`
    }
  }

  function startVideo(nonce: string, frameUrl: string) {
    stopAll()
    const iframe = document.createElement('iframe')
    iframe.src = frameUrl
    iframe.setAttribute('aria-hidden', 'true')
    iframe.tabIndex = -1
    iframe.title = 'SignSeek capture'
    // 1x1 and transparent, but inside the viewport (off-screen cross-origin frames get throttled)
    iframe.style.cssText =
      'position:fixed;right:0;bottom:0;width:1px;height:1px;opacity:0;border:0;pointer-events:none;z-index:-1'
    const channel = new MessageChannel()
    const port = channel.port1
    let ready = false
    let inFlight = false
    let stopped = false
    let badChecks = 0
    let canvas: OffscreenCanvas | null = null
    let rvfc = 0
    let lastCheck = 0
    let due = 0
    let video = pickVideo()

    const readyTimeout = setTimeout(() => {
      if (!ready) failed('capture iframe did not start (blocked by the page, or MediaPipe failed to load)')
    }, 10000)

    port.onmessage = (e: MessageEvent<PortMsg>) => {
      const m = e.data
      if (m.type === 'ready') {
        ready = true
        clearTimeout(readyTimeout)
        loop()
      } else if (m.type === 'done') {
        inFlight = false
      } else if (m.type === 'stop') {
        stopAll()
      } else if (m.type === 'error') {
        failed(`capture iframe error: ${m.error}`)
      }
    }

    const tick = () => {
      if (stopped) return
      const now = performance.now()
      if (!video || !video.isConnected || video.videoWidth === 0) video = pickVideo() // e.g. SPA navigation
      if (video && ready && !inFlight && now >= due && video.readyState >= 2) {
        const interval = 1000 / FPS
        due = due < now - interval ? now + interval : due + interval
        if (now - lastCheck > CHECK_EVERY_MS) {
          lastCheck = now
          const problem = check(video)
          badChecks = problem ? badChecks + 1 : 0
          if (problem === 'tainted' || (problem && badChecks >= 3)) {
            failed(`page video became ${problem}`)
            return
          }
        }
        const w = video.videoWidth
        const h = video.videoHeight
        if (!canvas || canvas.width !== w || canvas.height !== h) canvas = new OffscreenCanvas(w, h)
        canvas.getContext('2d')!.drawImage(video, 0, 0, w, h) // native resolution
        const bitmap = canvas.transferToImageBitmap()
        inFlight = true
        port.postMessage({ type: 'frame', bitmap, t: video.currentTime, width: w, height: h } satisfies PortMsg, [bitmap])
      }
      loop()
    }
    const loop = () => {
      if (stopped) return
      if (video && 'requestVideoFrameCallback' in video) rvfc = video.requestVideoFrameCallback(tick)
      else rvfc = requestAnimationFrame(tick)
    }

    iframe.addEventListener('load', () => {
      // The port goes only to our iframe, with a one-time nonce the page cannot read.
      iframe.contentWindow?.postMessage({ type: 'signseek-init', nonce }, new URL(frameUrl).origin, [channel.port2])
    })
    ;(document.body ?? document.documentElement).appendChild(iframe)

    stopFns.push(() => {
      stopped = true
      clearTimeout(readyTimeout)
      if (video && 'cancelVideoFrameCallback' in video) video.cancelVideoFrameCallback(rvfc)
      cancelAnimationFrame(rvfc)
      try {
        port.postMessage({ type: 'stop' } satisfies PortMsg)
      } catch {
        /* already closed */
      }
      port.close()
      iframe.remove()
    })
  }

  function trackRect() {
    stopAll()
    let last = ''
    const report = () => {
      const v = pickVideo()
      let crop: Crop | null = null
      if (v) {
        const r = v.getBoundingClientRect() // viewport-relative CSS px: scroll is already applied
        const vw = window.innerWidth
        const vh = window.innerHeight
        const x0 = Math.max(0, r.left) / vw
        const y0 = Math.max(0, r.top) / vh
        const x1 = Math.min(vw, r.right) / vw
        const y1 = Math.min(vh, r.bottom) / vh
        if (x1 > x0 && y1 > y0) crop = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
      }
      const key = crop ? [crop.x, crop.y, crop.w, crop.h].map((n) => n.toFixed(3)).join() : 'none'
      if (key === last) return
      last = key
      chrome.runtime.sendMessage({ target: 'pipeline', type: 'crop', crop }).catch(() => {})
    }
    report()
    const timer = setInterval(report, 250)
    window.addEventListener('scroll', report, { passive: true })
    window.addEventListener('resize', report)
    stopFns.push(() => {
      clearInterval(timer)
      window.removeEventListener('scroll', report)
      window.removeEventListener('resize', report)
    })
  }

  chrome.runtime.onMessage.addListener((msg: CsMsg) => {
    if (msg.cs === 'start-video') startVideo(msg.nonce, msg.frameUrl)
    else if (msg.cs === 'track-rect') trackRect()
    else if (msg.cs === 'stop') stopAll()
  })
})()
