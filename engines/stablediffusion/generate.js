'use strict';

/**
 * Text-to-image via a one-shot bundled sd-cli process.
 * Width, height, steps, CFG, and seed are omitted so the CLI / model defaults apply.
 * One image, current prompt only.
 */

const fs = require('fs');
const fsp = require('fs').promises;
const path = require('path');
const { isForceCpu, isNoCudaKernelImage, withCudaHidden } = require('../common/gpuRuntime');
const { runSdCli } = require('./cli');
const { hasSplitWeights, resolveRunComponents } = require('./weights');

/**
 * @param {{
 *   modelPath: string,
 *   prompt: string,
 *   outputPath: string,
 *   components?: {
 *     vae?: string | null,
 *     clipL?: string | null,
 *     clipG?: string | null,
 *     t5xxl?: string | null,
 *     llm?: string | null,
 *   } | null,
 *   diffusionOnly?: boolean,
 *   initImage?: string | null,
 *   forceCpu?: boolean,
 * }} opts
 * @returns {string[]}
 */
function buildSdCliArgs(opts) {
  const forceCpu = opts.forceCpu != null ? Boolean(opts.forceCpu) : isForceCpu();
  const components = opts.components || {};
  const args = [];
  // A standalone diffusion GGUF has no checkpoint version until it is loaded
  // with --diffusion-model. -m is only for a complete checkpoint.
  if (opts.diffusionOnly || hasSplitWeights(components)) {
    args.push('--diffusion-model', opts.modelPath);
    if (components.vae) {
      args.push('--vae', components.vae);
    }
    if (components.clipL) {
      args.push('--clip_l', components.clipL);
    }
    if (components.clipG) {
      args.push('--clip_g', components.clipG);
    }
    if (components.t5xxl) {
      args.push('--t5xxl', components.t5xxl);
    }
    if (components.llm) {
      args.push('--llm', components.llm);
    }
  } else {
    args.push('-m', opts.modelPath);
  }
  args.push(
    '--mode',
    'img_gen',
    '-p',
    opts.prompt,
    '-o',
    opts.outputPath,
  );
  if (typeof opts.initImage === 'string' && opts.initImage.trim()) {
    args.push('--init-img', opts.initImage);
  }
  if (forceCpu) {
    args.push('--backend', 'cpu');
  }
  return args;
}

/**
 * sd-cli logs CUDA/Vulkan init before the real failure. Prefer the error
 * lines and drop the per-tensor metadata spam.
 * @param {string} stderr
 * @param {string} stdout
 * @param {number | null} code
 * @returns {string}
 */
function formatSdCliError(stderr, stdout, code) {
  const lines = `${stderr || ''}\n${stdout || ''}`
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const useful = [];
  for (const line of lines) {
    if (/not in model metadata/i.test(line)) {
      continue;
    }
    if (/^(ggml_cuda_init|ggml_vulkan|load_backend):/i.test(line)) {
      continue;
    }
    if (!/\[ERROR\s*\]|error:/i.test(line)) {
      continue;
    }
    const text = line.replace(/^\[[A-Z]+\s*\]\s+\S+\s+-\s+/, '').trim();
    if (text && !useful.includes(text)) {
      useful.push(text);
    }
  }
  const message = useful.slice(0, 4).join(' ');
  if (message) {
    return message.length > 500 ? `${message.slice(0, 500)}…` : message;
  }
  const fallback = (stderr || stdout || '').replace(/\s+/g, ' ').trim();
  if (fallback.length > 400) {
    return `${fallback.slice(0, 400)}…`;
  }
  return fallback || `sd-cli failed (${code})`;
}

/**
 * @param {string} modelPath
 * @param {string} prompt
 * @param {string} outputPath
 * @param {{ signal?: AbortSignal, modelRoot?: string, modelsCacheDir?: string, initImage?: string | null }} [opts]
 * @returns {Promise<string>}
 */
async function generateImage(modelPath, prompt, outputPath, opts = {}) {
  const text = typeof prompt === 'string' ? prompt.trim() : '';
  if (!text) {
    throw new Error('Text-to-image requires a prompt.');
  }
  if (!outputPath) {
    throw new Error('Text-to-image output path is not configured.');
  }

  const modelRoot = opts.modelRoot || path.dirname(modelPath);
  const components = resolveRunComponents(modelRoot, opts.modelsCacheDir);

  const runOnce = async (weightArgs) => {
    let result = await runSdCli(weightArgs, { signal: opts.signal });
    if (
      result.code !== 0 &&
      !(opts.signal && opts.signal.aborted) &&
      isNoCudaKernelImage(`${result.stderr}\n${result.stdout}`)
    ) {
      process.stderr.write('[glaux] CUDA has no kernel image for this GPU; retrying on Vulkan.\n');
      result = await runSdCli(weightArgs, {
        signal: opts.signal,
        env: withCudaHidden(process.env),
      });
    }
    return result;
  };

  const result = await runOnce(
    buildSdCliArgs({
      modelPath,
      prompt: text,
      outputPath,
      components,
      initImage: opts.initImage,
    })
  );

  if (opts.signal && opts.signal.aborted) {
    return '';
  }

  if (result.code !== 0) {
    throw new Error(formatSdCliError(result.stderr, result.stdout, result.code));
  }

  try {
    const stats = await fsp.stat(outputPath);
    if (!stats.isFile() || stats.size <= 0) {
      throw new Error('sd-cli did not write an image.');
    }
  } catch (err) {
    if (err && err.code === 'ENOENT') {
      throw new Error('sd-cli did not write an image.');
    }
    if (err instanceof Error && /did not write/.test(err.message)) {
      throw err;
    }
    if (!fs.existsSync(outputPath)) {
      throw new Error('sd-cli did not write an image.');
    }
  }

  return outputPath;
}

module.exports = {
  buildSdCliArgs,
  formatSdCliError,
  generateImage,
};
