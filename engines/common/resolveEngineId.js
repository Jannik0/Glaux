'use strict';

/**
 * Map weight format + pipeline_tag to an engine id.
 * Weight detection stays in modelFormat; this is the second-axis router.
 */

const ASR_PIPELINE_TAG = 'automatic-speech-recognition';

/**
 * @param {'huggingface' | 'llamacpp' | null} format
 * @param {string | null | undefined} pipelineTag
 * @returns {'huggingface' | 'llamacpp' | 'transcribecpp' | null}
 */
function resolveEngineId(format, pipelineTag) {
  if (format === 'huggingface') {
    return 'huggingface';
  }
  if (format === 'llamacpp') {
    if (pipelineTag === ASR_PIPELINE_TAG) {
      return 'transcribecpp';
    }
    return 'llamacpp';
  }
  return null;
}

module.exports = {
  ASR_PIPELINE_TAG,
  resolveEngineId,
};
