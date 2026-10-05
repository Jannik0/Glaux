'use strict';

/**
 * Map weight format + pipeline_tag to an engine id.
 * Weight detection stays in modelFormat; this is the second-axis router.
 */

const ASR_PIPELINE_TAG = 'automatic-speech-recognition';
const TEXT_TO_IMAGE_PIPELINE_TAG = 'text-to-image';

/**
 * @param {'huggingface' | 'llamacpp' | null} format
 * @param {string | null | undefined} pipelineTag
 * @returns {'huggingface' | 'llamacpp' | 'transcribecpp' | 'stablediffusion' | null}
 */
function resolveEngineId(format, pipelineTag) {
  if (format === 'huggingface') {
    return 'huggingface';
  }
  if (format === 'llamacpp') {
    if (pipelineTag === ASR_PIPELINE_TAG) {
      return 'transcribecpp';
    }
    if (pipelineTag === TEXT_TO_IMAGE_PIPELINE_TAG) {
      return 'stablediffusion';
    }
    return 'llamacpp';
  }
  return null;
}

module.exports = {
  ASR_PIPELINE_TAG,
  TEXT_TO_IMAGE_PIPELINE_TAG,
  resolveEngineId,
};
