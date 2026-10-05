'use strict';

/**
 * Text-to-image via a one-shot bundled sd-cli process.
 * Width, height, steps, CFG, and seed are omitted so the CLI / model defaults apply.
 * One image, current prompt only.
 */

const fs = require('fs');
const fsp = require('fs').promises;
const { isForceCpu, isNoCudaKernelImage, withCudaHidden } = require('../common/gpuRuntime');
const { runSdCli } = require('./cli');

/**
 * @param {{
 *   modelPath: string,
 *   prompt: string,
 *   outputPath: string,
 *   forceCpu?: boolean,
 * }} opts
 * @returns {string[]}
 */
function buildSdCliArgs(opts) {
  const forceCpu = opts.forceCpu != null ? Boolean(opts.forceCpu) : isForceCpu();
  const args = [
    '-m',
    opts.modelPath,
    '--mode',
    'img_gen',
    '-p',
    opts.prompt,
    '-o',
    opts.outputPath,
  ];
  if (forceCpu) {
    args.push('--backend', 'cpu');
  }
  return args;
}

/**
 * @param {string} modelPath
 * @param {string} prompt
 * @param {string} outputPath
 * @param {{ signal?: AbortSignal }} [opts]
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

  const args = buildSdCliArgs({
    modelPath,
    prompt: text,
    outputPath,
  });

  let result = await runSdCli(args, { signal: opts.signal });

  if (
    result.code !== 0 &&
    !(opts.signal && opts.signal.aborted) &&
    isNoCudaKernelImage(`${result.stderr}\n${result.stdout}`)
  ) {
    process.stderr.write('[glaux] CUDA has no kernel image for this GPU; retrying on Vulkan.\n');
    result = await runSdCli(args, {
      signal: opts.signal,
      env: withCudaHidden(process.env),
    });
  }

  if (opts.signal && opts.signal.aborted) {
    return '';
  }

  if (result.code !== 0) {
    const errText = (result.stderr || result.stdout || '').replace(/\s+/g, ' ').trim();
    throw new Error(
      errText.length > 400
        ? `${errText.slice(0, 400)}…`
        : errText || `sd-cli failed (${result.code})`
    );
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
  generateImage,
};
