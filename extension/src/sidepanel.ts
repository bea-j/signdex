// Side panel UI. Reuses the web app's pipeline through the @shared alias.
import { FEATURES_PER_FRAME, SEQUENCE_LENGTH } from '@shared/features'

const results = document.querySelector<HTMLDivElement>('#results')
if (results) {
  results.textContent = `Shared code loaded. Model input: ${SEQUENCE_LENGTH} x ${FEATURES_PER_FRAME}.`
}
