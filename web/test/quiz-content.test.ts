import { describe, expect, it } from 'vitest'
import { interchangeable, isMeaningOrUsage } from '../src/app/quiz-content'

// Real questions the live Worker returned (Nov 2026), all about how signs are made.
const FORMATION: [string, string[]][] = [
  ['Which parameter differentiates the sign for CAT from the sign for DOG?', ['Handshape', 'Location', 'Movement', 'Palm Orientation']],
  ['Which handshape is used to sign WATER?', ['W handshape', 'B handshape', 'A handshape', '5 handshape']],
  ['When signing DOG, where is the secondary motion or snap typically performed?', ['At the chin', 'Against the chest']],
  ['What is the primary location for the sign DAD?', ['Chin', 'Forehead', 'Chest', 'Nose']],
  ['What type of non-manual marker typically accompanies the sign HAPPY?', ['A frown', 'A smile']],
  ['Which movement characterizes the sign SAD?', ['Hands moving outward in circles', 'Hands sliding down the face']],
  ['Where does the thumb make contact when signing MOM?', ['Chin', 'Forehead']],
  ['In the sign UP, what is the primary movement direction?', ['Downward', 'Upward']],
  ['Which parameter is most crucial for identifying the sign FEET?', ['Facial expression', 'Blinking']],
]

describe('quiz content', () => {
  it.each(FORMATION)('drops: %s', (question, options) => {
    expect(isMeaningOrUsage({ question, options })).toBe(false)
  })

  it.each([
    // shaped like a meaning question, but smuggles in formation or a comparison
    ['Which sign means the same handshape as DOG?', ['Cat', 'Bird']],
    ['What does CAT mean?', ['A pet', 'A tap on the cheek']],
    ['Which sign would you use that shares a movement with HAPPY?', ['Please', 'Sad']],
    // no formation word, but not a meaning or usage question either
    ['Which two signs are easy to mix up?', ['Cat and dog', 'Mom and dad']],
  ] as [string, string[]][])('drops disguised: %s', (question, options) => {
    expect(isMeaningOrUsage({ question, options })).toBe(false)
  })

  it.each([
    ['What does CAT mean?', ['A pet that meows', 'A pet that barks']],
    ['Which sign means “Mother”?', ['Mom', 'Dad']],
    ['Which sign would you use to ask for something politely?', ['Please', 'No']],
    ['When would you use THANK YOU?', ['After someone helps you', 'When you are lost']],
  ] as [string, string[]][])('keeps: %s', (question, options) => {
    expect(isMeaningOrUsage({ question, options })).toBe(true)
  })

  it('never offers two interchangeable answers', () => {
    expect(interchangeable('cut', 'scissors')).toBe(true)
    expect(interchangeable('kitty', 'cat')).toBe(true)
    expect(interchangeable('cat', 'dog')).toBe(false)
    expect(interchangeable('cat', 'cat')).toBe(false)
  })
})
