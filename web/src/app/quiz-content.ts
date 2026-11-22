// What a quiz question may say: what a sign means and when it is used, never how it is made (handshape, movement,
// location, orientation, facial expression) or how two signs compare. The quiz service's generated descriptions
// of signs have been wrong (DRINK with a fist; CAT and DOG "sharing a handshape"), and they can't be verified.
//
// A question from the service must be shaped like a meaning/usage question AND use no formation wording. It is an
// allowlist on purpose: an unexpected kind of question is dropped, and the dictionary fills the gap.
// The service's explanations are never shown (see quiz.ts).

const MEANING_OR_USAGE = [
  /^what does .{1,60} mean( in english)?\?$/i,
  /^what (is|’s|'s) the (english )?meaning of .{1,60}\?$/i,
  /^which (sign|word) means\b/i,
  /^which (sign|word) (would|do|could|should) you use\b/i,
  /^which (sign|word) (best )?(fits|completes)\b/i,
  /^when would you use\b/i,
  /^in (which|what) (situation|sentence|context)\b/i,
  /^which (situation|sentence)\b/i,
]

const FORMATION = new RegExp(
  '\\b(' +
    [
      // the parameters of a sign
      'hand ?-?shapes?', 'movements?', 'motions?', 'locations?', 'located', 'orientations?', 'parameters?', 'non-?manual',
      'facial expressions?', 'gestures?', 'articulat\\w*',
      // describing production
      'signed', 'signing', 'produced', 'performed', 'formed', 'mimic\\w*', 'tap(s|ped|ping)?', 'twist\\w*', 'wiggl\\w*',
      'strok\\w*', 'pinch\\w*', 'rub(s|bed|bing)?', 'flick\\w*', 'circling', 'circular', 'downward', 'upward',
      'hands?', 'palms?', 'fingers?', 'thumbs?', 'fists?', 'knuckles?', 'wrists?', 'fingertips?', '[a-z0-9][- ]hand',
      'forehead', 'chin', 'cheeks?', 'chest',
      // comparing two signs
      'differentiat\\w*', 'distinguish\\w*', 'differs?', 'contrast\\w*', 'shares?', 'sharing', 'look-?alikes?',
      'looks? (like|similar)', 'similar',
    ].join('|') +
    ')\\b',
  'i',
)

export const describesFormation = (text: string): boolean => FORMATION.test(text)

export function isMeaningOrUsage(q: { question: string; options: string[] }): boolean {
  return MEANING_OR_USAGE.some((re) => re.test(q.question.trim())) && ![q.question, ...q.options].some(describesFormation)
}

/** Labels that are the same sign, or the same meaning: they can't be right and wrong answers to one question. */
const INTERCHANGEABLE = [
  ['awake', 'wake'],
  ['cut', 'scissors'],
  ['lips', 'mouth'],
  ['stay', 'that'],
  ['that', 'there'],
  ['gift', 'give'],
  ['sleep', 'sleepy', 'nap', 'bed'],
  ['cat', 'kitty'],
  ['dog', 'puppy'],
  ['quiet', 'shhh'],
  ['hear', 'listen', 'ear'],
  ['look', 'see', 'eye'],
  ['loud', 'noisy'],
  ['food', 'snack'],
  ['close', 'closet'],
  ['bad', 'yucky'],
  ['dirty', 'yucky'],
  ['owie', 'sick'],
  ['wet', 'water'],
  ['bird', 'hen', 'duck', 'goose'],
  ['bedroom', 'room'],
  ['minemy', 'yourself'],
  ['pen', 'pencil'],
]

export function interchangeable(a: string, b: string): boolean {
  return a !== b && INTERCHANGEABLE.some((g) => g.includes(a) && g.includes(b))
}
