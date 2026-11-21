// The one feature spec shared with model/prepare.py. Never hard-code indices elsewhere.
import spec from '../../model/feature_spec.json'

export const featureSpec = spec
export type FeatureSpec = typeof spec
