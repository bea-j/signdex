// Quiz from "My signs". Questions come from the Signdex quiz Worker (only the saved signs' names are sent: no
// video, no landmarks). If the Worker is unreachable or returns something unusable, a local quiz is built from
// similar.json (look-alike signs as the wrong options), so the quiz also works offline.
import { referenceClip } from '../reference'
import { prefersReducedMotion, SkeletonPlayer } from '../ui/skeleton'
import { entry, knownLabels, loadReferenceData, signerLink, similarTo } from './engine'
import { onEnter, onLeave, show } from './router'
import { savedSigns } from './saved'

export const QUIZ_URL = 'https://asl-tutor-api.ph105.workers.dev/'
const MIN_SIGNS = 4
const MAX_LOCAL_QUESTIONS = 10
const TIMEOUT_MS = 25_000

export interface QuizQuestion {
  targetSign: string
  question: string
  options: string[]
  correctIndex: number
  explanation: string
  /** Our label for targetSign, when it maps to one (for the reference animation and the review list). */
  label?: string
}

/** Request body for the Worker. Only sign names. (Field name and name style to be confirmed with its owner.) */
export function workerRequest(displayNames: string[]): unknown {
  return { savedSigns: displayNames }
}

class QuizServiceError extends Error {}

/** Validate the Worker's JSON: { questions: [{ targetSign, question, options[], correctIndex, explanation }] }. */
export function parseQuiz(data: unknown): QuizQuestion[] {
  const qs = (data as { questions?: unknown })?.questions
  if (!Array.isArray(qs) || qs.length === 0) throw new QuizServiceError('no questions in the response')
  return qs.map((q, i) => {
    const o = q as Record<string, unknown>
    const options = o.options
    const ci = o.correctIndex
    if (typeof o.question !== 'string' || !o.question.trim()) throw new QuizServiceError(`question ${i + 1}: no text`)
    if (!Array.isArray(options) || options.length < 2 || !options.every((x) => typeof x === 'string'))
      throw new QuizServiceError(`question ${i + 1}: bad options`)
    if (typeof ci !== 'number' || !Number.isInteger(ci) || ci < 0 || ci >= options.length)
      throw new QuizServiceError(`question ${i + 1}: bad correctIndex`)
    return {
      targetSign: typeof o.targetSign === 'string' ? o.targetSign : String(options[ci]),
      question: o.question,
      options: options as string[],
      correctIndex: ci,
      explanation: typeof o.explanation === 'string' ? o.explanation : '',
    }
  })
}

async function fetchWorkerQuiz(labels: string[]): Promise<QuizQuestion[]> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS)
  try {
    const r = await fetch(QUIZ_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(workerRequest(labels.map((l) => entry(l).display!))),
      signal: ctrl.signal,
    })
    let data: unknown
    try {
      data = await r.json()
    } catch {
      throw new QuizServiceError('the quiz service sent something that is not JSON')
    }
    if (!r.ok) throw new QuizServiceError(`the quiz service answered ${r.status}`)
    return parseQuiz(data)
  } finally {
    clearTimeout(timer)
  }
}

// ---------------------------------------------------------------- local quiz from similar.json
const shuffle = <T>(a: T[]): T[] => {
  const b = [...a]
  for (let i = b.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[b[i], b[j]] = [b[j], b[i]]
  }
  return b
}

const listOf = (xs: string[]) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`)

export async function localQuiz(labels: string[]): Promise<QuizQuestion[]> {
  const all = knownLabels()
  const out: QuizQuestion[] = []
  for (const label of shuffle(labels).slice(0, MAX_LOCAL_QUESTIONS)) {
    const e = entry(label)
    const sims = similarTo(label, 3).map((s) => s.label).filter((l) => l !== label)
    const wrong = [...sims]
    for (const l of shuffle([...labels, ...all])) {
      if (wrong.length >= 3) break
      if (l !== label && !wrong.includes(l)) wrong.push(l)
    }
    const options = shuffle([label, ...wrong.slice(0, 3)])
    const hasClip = !!(await referenceClip(label))
    const look = sims.length ? ` It looks similar to ${listOf(sims.map((l) => entry(l).display!))}, so watch the handshape and movement.` : ''
    out.push({
      targetSign: e.display!,
      label,
      question: hasClip ? 'Which sign is this signer making?' : `Which sign means “${e.meaning ?? e.display}”?`,
      options: options.map((l) => entry(l).display!),
      correctIndex: options.indexOf(label),
      explanation: `${e.display}${e.meaning ? ` means “${e.meaning}”.` : '.'}${look}`,
    })
  }
  return out
}

// ---------------------------------------------------------------- UI
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const startBtn = $<HTMLButtonElement>('start-quiz')
const optionsEl = $('quiz-options')
const feedback = $('quiz-feedback')
const nextBtn = $<HTMLButtonElement>('quiz-next')
const refBtn = $<HTMLButtonElement>('quiz-ref')

let questions: QuizQuestion[] = []
let index = 0
let answers: boolean[] = []
let answered = false
let player: SkeletonPlayer | null = null
let run = 0

function updateStart() {
  const n = savedSigns.list().length
  startBtn.disabled = n < MIN_SIGNS
  $('quiz-hint').textContent =
    n < MIN_SIGNS
      ? `Save at least ${MIN_SIGNS} signs to start a quiz (${n} saved).`
      : `Multiple choice on your ${n} saved signs.`
}

function view(state: 'loading' | 'question' | 'summary') {
  $('quiz-loading').hidden = state !== 'loading'
  $('quiz-q').hidden = state !== 'question'
  $('quiz-summary').hidden = state !== 'summary'
}

/** Our label for a sign name the Worker sends back (display name or label, any case). */
function labelFor(name: string, saved: string[]): string | undefined {
  const n = name.trim().toLowerCase()
  return saved.find((l) => l.toLowerCase() === n || entry(l).display!.toLowerCase() === n)
}

export async function startQuiz() {
  const my = ++run
  const saved = savedSigns.list()
  if (saved.length < MIN_SIGNS) return
  show('quiz')
  view('loading')
  $('quiz-error').hidden = true
  $('quiz-progress').textContent = 'Quiz'
  $('quiz-source').textContent = ''
  await loadReferenceData()
  try {
    const qs = await fetchWorkerQuiz(saved)
    if (my !== run) return
    questions = qs.map((q) => ({ ...q, label: labelFor(q.targetSign, saved) }))
    $('quiz-source').textContent = 'Questions written by the Signdex quiz service'
  } catch (e) {
    if (my !== run) return
    console.warn('quiz service failed:', e)
    $('quiz-error-text').textContent = 'We couldn’t get questions from the quiz service, so here is an offline quiz built from look-alike signs.'
    $('quiz-error').hidden = false
    questions = await localQuiz(saved)
    if (my !== run) return
    $('quiz-source').textContent = 'Offline quiz'
  }
  begin()
}

function begin() {
  index = 0
  answers = []
  renderQuestion()
}

function renderQuestion() {
  const q = questions[index]
  answered = false
  view('question')
  $('quiz-progress').textContent = `Question ${index + 1} of ${questions.length}`
  $('quiz-question').textContent = q.question
  feedback.hidden = true
  nextBtn.hidden = true
  optionsEl.replaceChildren(
    ...q.options.map((opt, i) => {
      const li = document.createElement('li')
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'quiz-option'
      b.innerHTML = `<span class="key" aria-hidden="true">${i + 1}</span><span class="text"></span><span class="mark"></span>`
      b.querySelector('.text')!.textContent = opt
      b.addEventListener('click', () => choose(i))
      li.append(b)
      return li
    }),
  )
  // the target sign's reference animation, when there is one
  player?.pause()
  player = null
  refBtn.hidden = true
  if (q.label) {
    const my = run
    void referenceClip(q.label).then((clip) => {
      if (!clip || my !== run || questions[index] !== q) return
      refBtn.hidden = false
      player = new SkeletonPlayer(refBtn.querySelector('canvas')!, clip)
      if (prefersReducedMotion()) player.still()
      else player.play(0)
    })
  }
  $('quiz-question').focus()
}

function choose(i: number) {
  if (answered) return
  answered = true
  const q = questions[index]
  const right = i === q.correctIndex
  answers[index] = right
  optionsEl.querySelectorAll<HTMLButtonElement>('.quiz-option').forEach((b, j) => {
    b.disabled = true
    if (j === q.correctIndex) {
      b.classList.add('correct')
      b.querySelector('.mark')!.textContent = j === i ? 'Your answer ✓' : 'Correct answer'
    } else if (j === i) {
      b.classList.add('wrong')
      b.querySelector('.mark')!.textContent = 'Your answer'
    } else b.classList.add('dim')
  })
  feedback.className = `quiz-feedback ${right ? 'ok' : 'no'}`
  feedback.innerHTML = '<strong></strong><span></span>'
  feedback.querySelector('strong')!.textContent = right ? 'Correct!' : `Not quite. It’s ${q.options[q.correctIndex]}.`
  feedback.querySelector('span')!.textContent = q.explanation
  feedback.hidden = false
  nextBtn.hidden = false
  nextBtn.firstChild!.textContent = index === questions.length - 1 ? 'See results ' : 'Next '
  nextBtn.focus()
}

function summary() {
  player?.pause()
  view('summary')
  const n = questions.length
  const score = answers.filter(Boolean).length
  $('quiz-progress').textContent = 'Results'
  $('quiz-score').textContent = `${score} of ${n} correct`
  const missed = questions.filter((_, i) => !answers[i])
  $('quiz-score-note').textContent = missed.length ? 'Signs to review:' : 'Every sign right. Nice work!'
  const review = $('quiz-review')
  review.replaceChildren()
  if (missed.length) {
    const ul = document.createElement('ul')
    ul.className = 'saved-list'
    for (const q of missed) {
      const li = document.createElement('li')
      li.className = 'saved-item glass-panel'
      const name = document.createElement('span')
      name.className = 'sign-name'
      name.textContent = q.label ? entry(q.label).display! : q.targetSign
      li.append(name)
      if (q.label) {
        const link = signerLink(entry(q.label))
        const a = document.createElement('a')
        Object.assign(a, { className: 'link', href: link.href, target: '_blank', rel: 'noreferrer' })
        a.setAttribute('aria-label', link.aria)
        a.innerHTML = 'Watch a real signer <svg class="icon-sm" aria-hidden="true"><use href="#i-external"/></svg>'
        li.append(a)
      }
      ul.append(li)
    }
    review.append(ul)
  }
  $('quiz-score').focus()
}

nextBtn.addEventListener('click', () => {
  if (index < questions.length - 1) {
    index++
    renderQuestion()
  } else summary()
})
$('quiz-retake').addEventListener('click', () => {
  questions = shuffle(questions)
  begin()
})
$('quiz-retry').addEventListener('click', () => void startQuiz())
refBtn.addEventListener('click', () => player?.play(0))
startBtn.addEventListener('click', () => void startQuiz())

// keys 1-4 pick an option while a question is open
document.addEventListener('keydown', (e) => {
  if ($('screen-quiz').hidden || $('quiz-q').hidden || answered || e.ctrlKey || e.altKey || e.metaKey) return
  const n = Number(e.key)
  if (n >= 1 && n <= questions[index]?.options.length) {
    e.preventDefault()
    choose(n - 1)
  }
})

savedSigns.onChange(updateStart)
updateStart()
onEnter('saved', updateStart)
onLeave('quiz', () => {
  run++
  player?.pause()
})
