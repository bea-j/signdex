// sketch.js — dapple: interactive light background (no UI panel)
// Needs p5.js 1.x loaded first. Runs in p5 instance mode and is wrapped in an IIFE,
// so it defines no globals except window.signdexBg and won't collide with site code.
//
//   <script src="https://cdn.jsdelivr.net/npm/p5@1.11.1/lib/p5.min.js"></script>
//   <script src="sketch.js"></script>
//
// Mounts a fixed, full-viewport <canvas id="signdex-bg"> (z-index 0, pointer-events: none,
// so every click reaches the page). Site content must sit above it: give the main content
// wrapper position: relative; z-index: 1. The cursor carries a soft light and the forms
// drift away from it. No keyboard or click handlers.
//
// API (window.signdexBg):
//   signdexBg.pause()                 freeze on the current frame and stop rendering
//   signdexBg.resume()
//   signdexBg.set('exposure', 0.9)    change any setting live

(function () {
  const SETTINGS = {
    speed: 0.35,
    exposure: 0.71,
    scale: 2.97,
    dapple: 0.65,
    softness: 0.2,
    wind: 1.4,
    shafts: 0.52,
    filaments: 0.5,
    chroma: 0.8,
    mute: 0.2,
    grain: 0.45,
    cursor: 0.6,
    forms: 16,
    inkAlpha: 0.35,
    blend: 0,          // 0 screen · 1 add · 2 soft light · 3 multiply · 4 normal
    res: 0.5,          // light pass resolution (0.5 is plenty and much cheaper)
  };

  const PALETTE = {
    light: ['#ff7a1f', '#ffb347', '#d23c1e', '#2e9e83'],
    base: '#0c0b0a',
    ink: ['#ffb36b', '#7fd1b9', '#f25c3b', '#efe7d4'],
  };

  const VERT = `
attribute vec3 aPosition;
attribute vec2 aTexCoord;
varying vec2 vUv;
void main() {
  vUv = aTexCoord;
  vec4 p = vec4(aPosition, 1.0);
  p.xy = p.xy * 2.0 - 1.0;
  gl_Position = p;
}`;
  const NOISE = `
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  mat2 r = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 5; i++) { v += a * noise(p); p = r * p * 2.02 + 17.0; a *= 0.5; }
  return v;
}`;
  const LIGHT_FRAG = `
precision highp float;
varying vec2 vUv;
uniform vec2 uRes, uSeed, uMouse;
uniform float uTime, uScale, uSoft, uDapple, uWarp, uShafts, uFil, uChroma, uExposure, uCursor;
uniform vec3 uL0, uL1, uL2, uL3;
${NOISE}

vec3 lightColor(int i) {
  if (i == 0) return uL0;
  if (i == 1) return uL1;
  if (i == 2) return uL2;
  return uL3;
}

vec3 field(vec2 uv) {
  float asp = uRes.x / uRes.y;
  vec2 a = vec2((uv.x - 0.5) * asp, uv.y - 0.5);
  float t = uTime;
  vec3 E = vec3(0.0);

  // drifting pools of light
  for (int i = 0; i < 4; i++) {
    float fi = float(i);
    vec2 c = vec2(sin(t * (0.13 + 0.05 * fi) + fi * 2.1 + uSeed.x) * 0.45 * asp,
                  cos(t * (0.11 + 0.04 * fi) + fi * 1.7 + uSeed.y) * 0.38);
    float r = 0.3 + 0.12 * sin(t * 0.2 + fi * 3.0);
    float d = length(a - c);
    E += lightColor(i) * exp(-d * d / (r * r)) * (1.6 - 0.25 * fi);
  }

  // light carried by the cursor
  float dm = length(a - uMouse);
  E += mix(uL0, vec3(1.0, 0.95, 0.85), 0.5) * exp(-dm * dm / 0.025) * 1.8 * uCursor;

  // diagonal shafts
  float ang = -0.55;
  vec2 rp = mat2(cos(ang), -sin(ang), sin(ang), cos(ang)) * a;
  float s = fbm(vec2(rp.x * 3.0 + uSeed.x, rp.y * 0.35 - t * 0.04));
  E *= mix(1.0, 0.25 + 1.7 * smoothstep(0.3, 0.72, s), uShafts);

  // out-of-focus leaves between you and the light
  vec2 p = a * uScale + uSeed;
  vec2 q = vec2(fbm(p + vec2(0.0, t * 0.15)), fbm(p + vec2(5.2, 1.3) - t * 0.12));
  vec2 w = p + uWarp * q;
  float leaf = smoothstep(0.48 - uSoft, 0.48 + uSoft, fbm(w));
  E *= mix(1.0, 0.06 + 0.94 * leaf, uDapple);

  // thin refraction filaments, only visible where there is light
  float rdg = 1.0 - abs(2.0 * fbm(w * 0.7 + vec2(t * 0.05, 3.0)) - 1.0);
  E += E * pow(rdg, 14.0) * 2.5 * uFil;
  return E;
}

void main() {
  vec2 off = vec2(0.004, 0.002) * uChroma;
  vec3 E = vec3(field(vUv + off).r, field(vUv).g, field(vUv - off).b);
  gl_FragColor = vec4(1.0 - exp(-E * uExposure), 1.0);
}`;
  const OUT_FRAG = `
precision highp float;
varying vec2 vUv;
uniform sampler2D uLight, uForms;
uniform vec2 uRes;
uniform vec3 uBase;
uniform float uTime, uGrain, uMute, uMode;
${NOISE}

void main() {
  vec2 uv = vUv;
  vec3 L = texture2D(uLight, uv).rgb;
  vec3 col = uBase + L * (1.0 - uBase);

  vec4 F = texture2D(uForms, vec2(uv.x, 1.0 - uv.y));  // 2D canvas texture arrives flipped
  vec3 b;
  if (uMode < 0.5)      b = 1.0 - (1.0 - col) * (1.0 - F.rgb);
  else if (uMode < 1.5) b = col + F.rgb;
  else if (uMode < 2.5) b = mix(2.0 * col * F.rgb + col * col * (1.0 - 2.0 * F.rgb),
                                sqrt(col) * (2.0 * F.rgb - 1.0) + 2.0 * col * (1.0 - F.rgb),
                                step(0.5, F.rgb));
  else if (uMode < 3.5) b = col * F.rgb;
  else                  b = F.rgb;
  col = mix(col, b, F.a);

  float l = dot(col, vec3(0.299, 0.587, 0.114));
  col = mix(col, vec3(l), uMute * 0.6);
  col = col * (1.0 - uMute * 0.1) + uMute * 0.035;

  vec2 d = uv - 0.5;
  col *= 1.0 - dot(d, d) * 0.7;

  vec2 gp = uv * uRes + fract(uTime * 7.13) * 311.0;
  float g = (hash(gp) + hash(gp + 71.7)) * 0.5;
  col += (g - 0.5) * uGrain * 0.24;

  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}`;

  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const sketch = (p) => {
    let lightBuf, lightSh, outBuf, outSh, formLayer;
    let forms = [];
    let t = 0, paused = false;
    let seed = Math.random() * 100;
    let mouseL = [0, 0];
    let hasMouse = false;
    let baseRGB, lightRGB;

    const rgb = (hex) => {
      const c = p.color(hex);
      return [p.red(c) / 255, p.green(c) / 255, p.blue(c) / 255];
    };

    // ------------------------------------------------------------ forms
    class Form {
      constructor() { this.reset(); }

      reset() {
        this.kind = p.random(['blob', 'halo', 'rings', 'rings', 'scribble', 'dots']);
        this.x = p.random(p.width);
        this.y = p.random(p.height);
        this.r = p.random(0.06, 0.2) * Math.min(p.width, p.height);
        this.s = p.random(1000);
        this.ink = Math.floor(p.random(4));
        this.vx = 0; this.vy = 0;
      }

      update() {
        const sp = SETTINGS.speed;
        this.vx += (p.noise(this.s, t * 0.4) - 0.5) * 0.5 * sp;
        this.vy += (p.noise(this.s + 50, t * 0.4) - 0.5) * 0.5 * sp;
        if (hasMouse) {
          const dx = this.x - p.mouseX, dy = this.y - p.mouseY;
          const d = Math.sqrt(dx * dx + dy * dy), reach = this.r * 2.2;
          if (d < reach && d > 0.01) {
            const f = (1 - d / reach) * 1.4;
            this.vx += (dx / d) * f; this.vy += (dy / d) * f;
          }
        }
        this.vx *= 0.93; this.vy *= 0.93;
        this.x += this.vx; this.y += this.vy;
        const m = this.r * 2;
        if (this.x < -m) this.x = p.width + m;
        if (this.x > p.width + m) this.x = -m;
        if (this.y < -m) this.y = p.height + m;
        if (this.y > p.height + m) this.y = -m;
      }

      outline(r, amp = 0.35, k = 1.3) {
        const pts = [], N = 96;
        for (let i = 0; i < N; i++) {
          const a = (i / N) * p.TWO_PI;
          const n = p.noise(Math.cos(a) * k + this.s, Math.sin(a) * k + this.s, t * 0.5);
          const rr = r * (1 + amp * (n - 0.5) * 2);
          pts.push([this.x + Math.cos(a) * rr, this.y + Math.sin(a) * rr]);
        }
        return pts;
      }

      shape(g, pts) {
        g.beginShape();
        for (let i = 0; i < pts.length; i++) {
          g.vertex(pts[i][0], pts[i][1]);
        }
        g.endShape(p.CLOSE);
      }

      draw(g) {
        const c = p.color(PALETTE.ink[this.ink]);
        const a = SETTINGS.inkAlpha * 255;
        const kind = this.kind;
        g.push();
        if (kind === 'blob') {
          g.noStroke(); c.setAlpha(a); g.fill(c);
          this.shape(g, this.outline(this.r));
        } else if (kind === 'halo') {
          g.noStroke(); c.setAlpha(a * 0.7); g.fill(c);
          this.shape(g, this.outline(this.r, 0.25));
          const c2 = p.color(PALETTE.ink[(this.ink + 1) % 4]); c2.setAlpha(Math.min(255, a * 1.8));
          g.noFill(); g.stroke(c2); g.strokeWeight(1.5);
          g.translate(this.r * 0.08, -this.r * 0.06);
          this.shape(g, this.outline(this.r * 1.05, 0.3));
        } else if (kind === 'rings') {
          g.noFill(); c.setAlpha(Math.min(255, a * 1.6)); g.stroke(c); g.strokeWeight(1.1);
          for (let i = 0; i < 7; i++) {
            this.shape(g, this.outline(this.r * (1 - i * 0.12), 0.3 + i * 0.04));
          }
        } else if (kind === 'scribble') {
          g.noFill(); c.setAlpha(Math.min(255, a * 1.6)); g.stroke(c); g.strokeWeight(1.6); g.strokeJoin(p.ROUND);
          let x = this.x, y = this.y;
          g.beginShape();
          for (let i = 0; i < 140; i++) {
            g.vertex(x, y);
            const ang = p.noise(x * 0.004 + this.s, y * 0.004, t * 0.3) * p.TWO_PI * 2;
            x += Math.cos(ang) * 4; y += Math.sin(ang) * 4;
          }
          g.endShape();
        } else if (kind === 'dots') {
          g.noStroke(); c.setAlpha(Math.min(255, a * 1.4)); g.fill(c);
          const step = Math.max(10, this.r * 0.16);
          for (let gx = -this.r; gx <= this.r; gx += step) {
            for (let gy = -this.r; gy <= this.r; gy += step) {
              if (gx * gx + gy * gy <= this.r * this.r) {
                g.circle(this.x + gx, this.y + gy, step * 0.8 * p.noise(gx * 0.02 + this.s, gy * 0.02, t * 0.6));
              }
            }
          }
        }
        g.pop();
      }
    }

    // ------------------------------------------------------------ setup
    function build() {
      [lightBuf, outBuf, formLayer].forEach(g => g && g.remove());
      lightBuf = p.createGraphics(Math.ceil(p.width * SETTINGS.res), Math.ceil(p.height * SETTINGS.res), p.WEBGL);
      lightBuf.pixelDensity(1); lightBuf.noStroke();
      lightSh = lightBuf.createShader(VERT, LIGHT_FRAG);
      outBuf = p.createGraphics(p.width, p.height, p.WEBGL);
      outBuf.pixelDensity(1); outBuf.noStroke();
      outSh = outBuf.createShader(VERT, OUT_FRAG);
      formLayer = p.createGraphics(p.width, p.height);
      formLayer.pixelDensity(1);
      forms = [];
      for (let i = 0; i < SETTINGS.forms; i++) {
        forms.push(new Form());
      }
    }

    p.setup = () => {
      p.pixelDensity(1);
      const cnv = p.createCanvas(window.innerWidth, window.innerHeight);
      cnv.id('signdex-bg');
      cnv.elt.setAttribute('aria-hidden', 'true');
      cnv.style('position', 'fixed');
      cnv.style('inset', '0');
      cnv.style('width', '100vw');
      cnv.style('height', '100vh');
      cnv.style('z-index', '0');
      cnv.style('pointer-events', 'none');
      baseRGB = rgb(PALETTE.base);
      lightRGB = PALETTE.light.map(rgb);
      build();
      // start the light somewhere pleasant before the cursor arrives
      mouseL = [0.25 * p.width / p.height, -0.15];
      if (reduceMotion) { t = 12; p.noLoop(); p.redraw(); }
    };

    p.draw = () => {
      if (!paused && !reduceMotion) t += (p.deltaTime / 1000) * SETTINGS.speed;

      if (hasMouse) {
        const asp = p.width / p.height;
        mouseL[0] += ((p.mouseX / p.width - 0.5) * asp - mouseL[0]) * 0.08;
        mouseL[1] += ((p.mouseY / p.height - 0.5) - mouseL[1]) * 0.08;
      }

      lightBuf.shader(lightSh);
      lightSh.setUniform('uRes', [lightBuf.width, lightBuf.height]);
      lightSh.setUniform('uSeed', [seed, seed * 0.7]);
      lightSh.setUniform('uMouse', mouseL);
      lightSh.setUniform('uTime', t);
      lightSh.setUniform('uScale', SETTINGS.scale);
      lightSh.setUniform('uSoft', SETTINGS.softness);
      lightSh.setUniform('uDapple', SETTINGS.dapple);
      lightSh.setUniform('uWarp', SETTINGS.wind);
      lightSh.setUniform('uShafts', SETTINGS.shafts);
      lightSh.setUniform('uFil', SETTINGS.filaments);
      lightSh.setUniform('uChroma', SETTINGS.chroma);
      lightSh.setUniform('uExposure', SETTINGS.exposure);
      lightSh.setUniform('uCursor', hasMouse ? SETTINGS.cursor : 0);
      for (let i = 0; i < 4; i++) {
        lightSh.setUniform('uL' + i, lightRGB[i]);
      }
      lightBuf.rect(0, 0, lightBuf.width, lightBuf.height);

      formLayer.clear();
      for (const f of forms) {
        if (!paused && !reduceMotion) f.update();
        f.draw(formLayer);
      }

      outBuf.shader(outSh);
      outSh.setUniform('uLight', lightBuf);
      outSh.setUniform('uForms', formLayer);
      outSh.setUniform('uRes', [p.width, p.height]);
      outSh.setUniform('uBase', baseRGB);
      outSh.setUniform('uTime', t + p.frameCount * 0.001);
      outSh.setUniform('uGrain', SETTINGS.grain);
      outSh.setUniform('uMute', SETTINGS.mute);
      outSh.setUniform('uMode', SETTINGS.blend);
      outBuf.rect(0, 0, p.width, p.height);

      p.image(outBuf, 0, 0);
    };

    p.mouseMoved = () => { hasMouse = true; };
    p.mouseDragged = () => { hasMouse = true; };
    p.touchMoved = () => { hasMouse = true; };   // do not return false: page scrolling must keep working

    p.windowResized = () => {
      p.resizeCanvas(window.innerWidth, window.innerHeight);
      build();
      if (reduceMotion) p.redraw();
    };

    // ------------------------------------------------------------ api
    window.signdexBg = {
      pause() { paused = true; p.noLoop(); },
      resume() { if (!reduceMotion) { paused = false; p.loop(); } },
      set(key, value) {
        if (!(key in SETTINGS)) return;
        SETTINGS[key] = value;
        if (key === 'forms' || key === 'res') build();
        if (paused || reduceMotion) p.redraw();
      },
      settings: SETTINGS,
    };
  };

  const start = () => new p5(sketch, document.body);
  if (document.body) start();
  else document.addEventListener('DOMContentLoaded', start);
})();
