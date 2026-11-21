import { FEATURES_PER_FRAME, SEQUENCE_LENGTH } from './features'

const results = document.querySelector<HTMLDivElement>('#results')
if (results) {
  results.textContent = `Scaffold ready. Model input: ${SEQUENCE_LENGTH} frames x ${FEATURES_PER_FRAME} features.`
}
