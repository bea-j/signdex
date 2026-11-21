// Camera picker: lists video inputs (enumerateDevices), opens the chosen one, remembers it in localStorage.
// Reusable by the debug page and the main app: bind a <select>, then call open() to get a stream.

const STORAGE_KEY = 'signseek.cameraId'
/**
 * 1280x720, 16:9. resizeMode "none" asks for a native camera mode instead of Chrome cropping/scaling another
 * mode to fit, so wide cameras (e.g. Logitech Brio) deliver their full 16:9 field of view. All "ideal", so a
 * camera without that mode still opens with its closest one. The aspect difference from the ~3:4 training
 * video is corrected later by toTrainingAspect().
 */
const DEFAULT_CONSTRAINTS = {
  width: { ideal: 1280 },
  height: { ideal: 720 },
  aspectRatio: { ideal: 16 / 9 },
  resizeMode: { ideal: 'none' },
} as MediaTrackConstraints

/** localStorage can be unavailable (private mode, blocked site data): never let that break the page. */
function load(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

function save(id: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, id)
  } catch {
    /* ignore */
  }
}

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return []
  return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput')
}

export class CameraPicker {
  private cameras: MediaDeviceInfo[] = []
  private stream: MediaStream | null = null
  /** Called after the user picks a different camera while one is open, with the new stream. */
  onSwitch?: (stream: MediaStream, label: string) => void
  onError?: (message: string) => void

  constructor(private readonly select: HTMLSelectElement) {
    select.addEventListener('change', () => void this.picked())
    navigator.mediaDevices?.addEventListener?.('devicechange', () => void this.refresh())
  }

  /** The deviceId to open: the dropdown's value, else the remembered one. Empty string = browser default. */
  get selectedId(): string {
    return this.select.value || load() || ''
  }

  get label(): string {
    return this.select.selectedOptions[0]?.textContent ?? 'webcam'
  }

  /** Fill the dropdown. Labels are empty until camera permission is granted, so call again after open(). */
  async refresh(): Promise<void> {
    this.cameras = await listCameras()
    const want = this.stream?.getVideoTracks()[0]?.getSettings().deviceId ?? this.select.value ?? load() ?? ''
    this.select.innerHTML = ''
    if (this.cameras.length === 0) {
      this.select.append(new Option('No camera found', ''))
      this.select.disabled = true
      return
    }
    this.select.disabled = false
    this.cameras.forEach((c, i) => this.select.append(new Option(c.label || `Camera ${i + 1}`, c.deviceId)))
    const saved = this.cameras.find((c) => c.deviceId === want) ?? this.cameras.find((c) => c.deviceId === load())
    if (saved) this.select.value = saved.deviceId
  }

  /** Open the selected (or remembered) camera, falling back to the default one if it is gone. */
  async open(): Promise<MediaStream> {
    this.close()
    const id = this.selectedId
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: id ? { ...DEFAULT_CONSTRAINTS, deviceId: { exact: id } } : DEFAULT_CONSTRAINTS,
        audio: false,
      })
    } catch (e) {
      const err = e as DOMException
      if (!id || (err.name !== 'OverconstrainedError' && err.name !== 'NotFoundError')) throw e
      // remembered camera unplugged or renamed: use the default camera instead
      stream = await navigator.mediaDevices.getUserMedia({ video: DEFAULT_CONSTRAINTS, audio: false })
    }
    this.stream = stream
    await this.refresh() // permission granted now: real labels, and select the camera actually opened
    const actual = stream.getVideoTracks()[0]?.getSettings().deviceId
    if (actual) {
      this.select.value = actual
      save(actual)
    }
    return stream
  }

  close(): void {
    this.stream?.getTracks().forEach((t) => t.stop())
    this.stream = null
  }

  get isOpen(): boolean {
    return !!this.stream
  }

  /** What the camera actually delivers, e.g. "1280 × 720 @ 30 fps (16:9)". */
  get settingsText(): string {
    const s = this.stream?.getVideoTracks()[0]?.getSettings()
    if (!s?.width || !s.height) return ''
    const r = s.width / s.height
    const aspect = Math.abs(r - 16 / 9) < 0.02 ? '16:9' : Math.abs(r - 4 / 3) < 0.02 ? '4:3' : r.toFixed(2)
    return `${s.width} × ${s.height}${s.frameRate ? ` @ ${Math.round(s.frameRate)} fps` : ''} (${aspect})`
  }

  private async picked(): Promise<void> {
    save(this.select.value)
    if (!this.stream) return // not running: just remember the choice for the next open()
    try {
      const stream = await this.open()
      this.onSwitch?.(stream, this.label)
    } catch (e) {
      this.onError?.(`Could not open ${this.label}: ${(e as Error).message}`)
    }
  }
}
