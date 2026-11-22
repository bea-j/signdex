// Quiz from "My signs", about meanings and usage only (quiz-content.ts says what that allows).
// The Worker is asked for questions (only the saved signs' names are sent). Its meaning/usage questions are kept,
// always with the dictionary's meaning as the explanation; signs left without a question get one from the
// dictionary. If the Worker can't be reached, the whole quiz comes from the dictionary, so it also works offline.
import { referenceClip } from '../reference'
import { prefersReducedMotion, SkeletonPlayer } from '../ui/skeleton'
import { entry, knownLabels, loadReferenceData, signerLink, similarTo } from './engine'
import { interchangeable, isMeaningOrUsage } from './quiz-content'
import { onEnter, onLeave, show } from './router'
import { savedSigns } from './saved'

export const QUIZ_URL = 'https://asl-tutor-api.ph105.workers.dev/'
const MIN_SIGNS = 4
/** The Worker takes 2 to 15 signs (it slices anything over 15); we send the 15 most recently saved. */
const MAX_WORKER_SIGNS = 15
const MAX_QUESTIONS = 10
/**
 * Signs the quiz service is not asked about, and whose generated questions are dropped, because it produced
 * wrong content for them (DRINK: it claimed an S/fist handshape; DRINK uses a C handshape, like holding a cup).
 * They stay in My signs and in the offline quiz.
 */
const WORKER_EXCLUDED_SIGNS = ['drink']
const excluded = (s: string) => WORKER_EXCLUDED_SIGNS.some((x) => new RegExp(`(^|[^a-z])${x}([^a-z]|$)`, 'i').test(s))
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

/**
 * Request body for the Worker (format confirmed by its owner): {"savedWords": [sign names]}, 2 to 15 names
 * (display names or labels both work). `names` is in save order, oldest first: the 15 most recent are sent.
 */
export function workerRequest(names: string[]): { savedWords: string[] } {
  return { savedWords: names.slice(-MAX_WORKER_SIGNS) }
}

/** The Worker answered, but with an error or unusable JSON (no offline fallback for this). */
class QuizServiceError extends Error {}
/** The Worker could not be reached at all (offline, DNS, timeout): the offline quiz is used. */
class QuizUnreachableError extends Error {}

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
    let r: Response
    try {
      r = await fetch(QUIZ_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(workerRequest(labels.filter((l) => !excluded(l)).map((l) => entry(l).display!))),
        signal: ctrl.signal,
      })
    } catch (e) {
      throw new QuizUnreachableError((e as Error).name === 'AbortError' ? 'timed out' : (e as Error).message)
    }
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

// ---------------------------------------------------------------- dictionary questions
const shuffle = <T>(a: T[]): T[] => {
  const b = [...a]
  for (let i = b.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[b[i], b[j]] = [b[j], b[i]]
  }
  return b
}

/** The only explanation ever shown: the meaning from our dictionary. */
function explain(label: string): string {
  const e = entry(label)
  return e.meaning ? `${e.display} means “${e.meaning}”.` : ''
}

/** One meaning question per sign; signs the model confuses with it are the wrong options. */
export async function dictionaryQuiz(labels: string[], limit = MAX_QUESTIONS): Promise<QuizQuestion[]> {
  const all = knownLabels()
  const out: QuizQuestion[] = []
  for (const label of shuffle(labels).slice(0, limit)) {
    const e = entry(label)
    const wrong: string[] = []
    for (const l of [...similarTo(label, 3).map((s) => s.label), ...shuffle([...labels, ...all])]) {
      if (wrong.length >= 3) break
      // two options that mean the same thing (or are the same sign) would make the question unfair
      if (l !== label && !wrong.includes(l) && ![label, ...wrong].some((o) => interchangeable(o, l))) wrong.push(l)
    }
    const options = shuffle([label, ...wrong.slice(0, 3)])
    const hasClip = !!(await referenceClip(label))
    out.push({
      targetSign: e.display!,
      label,
      question: hasClip ? 'What does this sign mean?' : `Which sign means “${e.meaning ?? e.display}”?`,
      options: options.map((l) => entry(l).display!),
      correctIndex: options.indexOf(label),
      explanation: explain(label),
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
      : n > MAX_WORKER_SIGNS
        ? `Multiple choice on your ${MAX_WORKER_SIGNS} most recently saved signs.`
        : `Multiple choice on your ${n} saved signs.`
}

function view(state: 'loading' | 'question' | 'summary' | 'error') {
  $('quiz-loading').hidden = state !== 'loading'
  $('quiz-q').hidden = state !== 'question'
  $('quiz-summary').hidden = state !== 'summary'
}

/** Our label for a sign name the Worker sends back (display name or label, any case). */
function labelFor(name: string, saved: string[]): string | undefined {
  const key = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '')
  return saved.find((l) => key(l) === key(name) || key(entry(l).display!) === key(name))
}

/** The service's meaning/usage questions, plus dictionary questions for the signs those don't cover. */
async function composeQuiz(fromService: QuizQuestion[], saved: string[]) {
  const asked = saved.slice(-MAX_WORKER_SIGNS).filter((l) => !excluded(l))
  const kept: QuizQuestion[] = []
  for (const q of fromService) {
    if (kept.length >= MAX_QUESTIONS || !isMeaningOrUsage(q)) continue
    if ([q.targetSign, q.question, ...q.options].some(excluded)) continue
    const label = labelFor(q.targetSign, asked)
    if (!label) continue // not one of their signs
    kept.push({ ...q, label, explanation: explain(label) })
  }
  const uncovered = asked.filter((l) => !kept.some((q) => q.label === l))
  const filled = await dictionaryQuiz(uncovered, MAX_QUESTIONS - kept.length)
  const source = !kept.length
    ? 'Questions from the Signdex dictionary'
    : filled.length
      ? 'Questions from the quiz service and the Signdex dictionary'
      : 'Questions written by the Signdex quiz service'
  return { questions: [...kept, ...filled], source }
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
    const composed = await composeQuiz(qs, saved)
    if (my !== run) return
    questions = composed.questions
    $('quiz-source').textContent = composed.source
  } catch (e) {
    if (my !== run) return
    console.warn('quiz service failed:', e)
    $('quiz-error').hidden = false
    if (!(e instanceof QuizUnreachableError)) {
      // reachable but broken: say so, offer Try again (no offline quiz in this case)
      $('quiz-error-text').textContent = 'The quiz service had a problem making your questions. Please try again in a moment.'
      view('error')
      return
    }
    $('quiz-error-text').textContent = 'We couldn’t reach the quiz service, so here is an offline quiz from the Signdex dictionary.'
    questions = await dictionaryQuiz(saved)
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
