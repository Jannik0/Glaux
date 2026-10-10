'use strict';

/**
 * Text-to-image via a one-shot bundled sd-cli process.
 * Steps and CFG are omitted so the CLI / model defaults apply.
 * --seed -1 asks sd-cli to draw a seed. The CLI default of 42 would
 * otherwise repeat the same image for the same prompt.
 * An attached image would otherwise adopt the file's pixel size. That canvas is
 * fit back to the CLI default (512 on the long side) so the denoiser matches a
 * prompt-only run. Text-to-image passes the file as a reference image. FLUX and
 * Qwen Image edit from that reference. Z-Image cannot: this build concatenates
 * a reference latent and then builds positions without it, so sd-cli aborts.
 * Z-Image receives the file as --init-img at noise level 0.4. Image-to-image
 * keeps --init-img at the CLI strength. One image, current prompt only.
 */

const fs = require('fs');
const fsp = require('fs').promises;
const path = require('path');
const { isForceCpu, isNoCudaKernelImage, withCudaHidden } = require('../common/gpuRuntime');
const { runSdCli } = require('./cli');
const { hasSplitWeights, resolveRunComponents } = require('./weights');

/** sd.cpp identifies Z-Image by this diffusion tensor. */
const Z_IMAGE_TENSOR = 'cap_embedder.0.weight';
/**
 * Flow img2img mixes the latent as image * (1 - sigma) + noise * sigma.
 * The CLI default strength of 0.75 is 75% noise, so Z-Image draws a new picture.
 * 0.4 keeps the attachment and still lets the prompt change it.
 */
const Z_IMAGE_NOISE_LEVEL = '0.4';
/** Header plus tensor names. The weight blob starts after this. */
const Z_IMAGE_SCAN_BYTES = 16 * 1024 * 1024;

/** sd-cli draws a seed when the value is negative. 42 is its fixed default. */
const RANDOM_SEED = '-1';
/** sd-cli's default canvas when -W/-H are omitted. */
const INIT_CANVAS = 512;
/** Shared by SD (8), SDXL (8), and FLUX (16). */
const INIT_ALIGN = 64;

/**
 * Fit an init image inside the prompt-only canvas, preserving aspect ratio.
 * @param {number} width
 * @param {number} height
 * @returns {{ width: number, height: number }}
 */
function fitInitCanvas(width, height) {
  const w0 = Number(width);
  const h0 = Number(height);
  if (!Number.isFinite(w0) || !Number.isFinite(h0) || w0 < 1 || h0 < 1) {
    return { width: INIT_CANVAS, height: INIT_CANVAS };
  }
  const scale = Math.min(1, INIT_CANVAS / w0, INIT_CANVAS / h0);
  const align = (n) => Math.max(INIT_ALIGN, Math.floor((n * scale) / INIT_ALIGN) * INIT_ALIGN);
  return { width: align(w0), height: align(h0) };
}

/**
 * @param {Buffer} buf
 * @returns {{ width: number, height: number } | null}
 */
function imageSizeFromBuffer(buf) {
  if (!Buffer.isBuffer(buf) || buf.length < 10) {
    return null;
  }
  if (buf[0] === 0x89 && buf.toString('ascii', 1, 4) === 'PNG' && buf.length >= 24) {
    return positiveSize(buf.readUInt32BE(16), buf.readUInt32BE(20));
  }
  if (buf.toString('ascii', 0, 3) === 'GIF' && buf.length >= 10) {
    return positiveSize(buf.readUInt16LE(6), buf.readUInt16LE(8));
  }
  if (buf[0] === 0x42 && buf[1] === 0x4d && buf.length >= 26) {
    const header = buf.readUInt32LE(14);
    if (header === 12 && buf.length >= 22) {
      return positiveSize(buf.readUInt16LE(18), buf.readUInt16LE(20));
    }
    return positiveSize(buf.readInt32LE(18), Math.abs(buf.readInt32LE(22)));
  }
  if (buf.toString('ascii', 0, 4) === 'RIFF' && buf.toString('ascii', 8, 12) === 'WEBP') {
    return webpSize(buf);
  }
  if (buf[0] === 0xff && buf[1] === 0xd8) {
    return jpegSize(buf);
  }
  return null;
}

/**
 * @param {number} width
 * @param {number} height
 * @returns {{ width: number, height: number } | null}
 */
function positiveSize(width, height) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width < 1 || height < 1) {
    return null;
  }
  return { width, height };
}

/**
 * @param {Buffer} buf
 * @returns {{ width: number, height: number } | null}
 */
function webpSize(buf) {
  if (buf.length < 30) {
    return null;
  }
  const kind = buf.toString('ascii', 12, 16);
  if (kind === 'VP8X') {
    return positiveSize(1 + buf.readUIntLE(24, 3), 1 + buf.readUIntLE(27, 3));
  }
  if (kind === 'VP8 ') {
    return positiveSize(buf.readUInt16LE(26) & 0x3fff, buf.readUInt16LE(28) & 0x3fff);
  }
  if (kind === 'VP8L' && buf.length >= 25 && buf[20] === 0x2f) {
    const b0 = buf[21];
    const b1 = buf[22];
    const b2 = buf[23];
    const b3 = buf[24];
    const width = 1 + (((b1 & 0x3f) << 8) | b0);
    const height = 1 + (((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6));
    return positiveSize(width, height);
  }
  return null;
}

/**
 * @param {Buffer} buf
 * @returns {{ width: number, height: number } | null}
 */
function jpegSize(buf) {
  let i = 2;
  let size = null;
  let orientation = null;
  while (i + 8 < buf.length) {
    if (buf[i] !== 0xff) {
      i += 1;
      continue;
    }
    const marker = buf[i + 1];
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) {
      i += 2;
      continue;
    }
    if (i + 3 >= buf.length) {
      return null;
    }
    const len = buf.readUInt16BE(i + 2);
    if (len < 2) {
      return null;
    }
    if (marker === 0xe1 && orientation == null) {
      orientation = exifOrientation(buf, i + 4, i + 2 + len);
    }
    // SOF0–SOF15, except DHT (C4), JPG (C8), and DAC (CC).
    if (isJpegStartOfFrame(marker)) {
      if (i + 8 >= buf.length) {
        return null;
      }
      size = positiveSize(buf.readUInt16BE(i + 7), buf.readUInt16BE(i + 5));
      break;
    }
    i += 2 + len;
  }
  if (!size) {
    return null;
  }
  // Orientations 5–8 rotate the stored samples by 90 degrees.
  if (orientation != null && orientation >= 5 && orientation <= 8) {
    return { width: size.height, height: size.width };
  }
  return size;
}

/**
 * @param {number} marker
 * @returns {boolean}
 */
function isJpegStartOfFrame(marker) {
  return marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
}

/**
 * Orientation tag in an APP1 Exif segment, when the IFD0 entry is easy to read.
 * @param {Buffer} buf
 * @param {number} start
 * @param {number} end
 * @returns {number | null}
 */
function exifOrientation(buf, start, end) {
  if (start < 0 || end > buf.length || end - start < 16) {
    return null;
  }
  if (buf.toString('ascii', start, start + 4) !== 'Exif') {
    return null;
  }
  const tiff = start + 6;
  if (tiff + 8 > end) {
    return null;
  }
  const little = buf.toString('ascii', tiff, tiff + 2) === 'II';
  const big = buf.toString('ascii', tiff, tiff + 2) === 'MM';
  if (!little && !big) {
    return null;
  }
  const u16 = (offset) => (little ? buf.readUInt16LE(offset) : buf.readUInt16BE(offset));
  const u32 = (offset) => (little ? buf.readUInt32LE(offset) : buf.readUInt32BE(offset));
  if (u16(tiff + 2) !== 0x002a) {
    return null;
  }
  const ifd = tiff + u32(tiff + 4);
  if (ifd < tiff || ifd + 2 > end) {
    return null;
  }
  const count = u16(ifd);
  for (let n = 0; n < count; n += 1) {
    const entry = ifd + 2 + n * 12;
    if (entry + 12 > end) {
      return null;
    }
    if (u16(entry) !== 0x0112) {
      continue;
    }
    const type = u16(entry + 2);
    const values = u32(entry + 4);
    if (type !== 3 || values !== 1) {
      return null;
    }
    return u16(entry + 8);
  }
  return null;
}

/**
 * @param {string} filePath
 * @returns {{ width: number, height: number } | null}
 */
function readImageSize(filePath) {
  let fd = null;
  try {
    fd = fs.openSync(filePath, 'r');
    const buf = Buffer.alloc(512 * 1024);
    const n = fs.readSync(fd, buf, 0, buf.length, 0);
    return imageSizeFromBuffer(buf.subarray(0, n));
  } catch {
    return null;
  } finally {
    if (fd != null) {
      try {
        fs.closeSync(fd);
      } catch {
        // The size is optional; a missing header falls back to 512x512.
      }
    }
  }
}

/**
 * Z-Image reference images abort sd-cli (GGML_ASSERT on the position table).
 * The repo folder or the GGUF tensor name is enough to tell it from FLUX.
 * @param {string} modelPath
 * @returns {boolean}
 */
function isZImageModel(modelPath) {
  if (typeof modelPath !== 'string' || !modelPath.trim()) {
    return false;
  }
  if (/z[-_ ]?image/i.test(modelPath)) {
    return true;
  }
  let fd = null;
  try {
    fd = fs.openSync(modelPath, 'r');
    const buf = Buffer.alloc(Z_IMAGE_SCAN_BYTES);
    const n = fs.readSync(fd, buf, 0, buf.length, 0);
    return buf.subarray(0, n).includes(Z_IMAGE_TENSOR);
  } catch {
    return false;
  } finally {
    if (fd != null) {
      try {
        fs.closeSync(fd);
      } catch {
        // A missing model file is not Z-Image.
      }
    }
  }
}

function attachedImage(opts) {
  if (typeof opts.referenceImage === 'string' && opts.referenceImage.trim()) {
    return { path: opts.referenceImage, mode: 'reference' };
  }
  if (typeof opts.initImage === 'string' && opts.initImage.trim()) {
    return { path: opts.initImage, mode: 'init' };
  }
  return null;
}

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
 *   referenceImage?: string | null,
 *   width?: number,
 *   height?: number,
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
    '--seed',
    RANDOM_SEED,
  );
  const attached = attachedImage(opts);
  if (attached) {
    const canvas = fitInitCanvas(opts.width, opts.height);
    const zImage = isZImageModel(opts.modelPath);
    if (attached.mode === 'reference' && !zImage) {
      // --init-img noises the latent at strength 0.75. On a flow schedule that
      // is nearly pure noise, so FLUX draws a new image. A reference latent is
      // what these models actually edit from.
      args.push('--ref-image', attached.path);
    } else {
      args.push('--init-img', attached.path);
      if (zImage) {
        args.push('--strength', Z_IMAGE_NOISE_LEVEL);
        args.push('--extra-sample-args', 'strength_as_noise_level=true,force_first_sigma=true');
      }
    }
    // The CLI otherwise sets the canvas to the file's pixel size. A full-resolution
    // FLUX double block then fails while preparing weights.
    args.push('--width', String(canvas.width), '--height', String(canvas.height));
    // Decode retries a full-frame VAE graph with tiling when it does not fit.
    // Encode does not, so the same graph fails while preparing weights.
    args.push('--vae-tiling');
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
    if (isBackendInitLine(line)) {
      continue;
    }
    if (!/\[ERROR\s*\]|error:|GGML_ASSERT|Assertion |cudaMalloc failed|out of memory/i.test(line)) {
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
  const fallback = lines
    .filter((line) => !isBackendInitLine(line) && !/not in model metadata/i.test(line))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (fallback.length > 400) {
    return `…${fallback.slice(-400)}`;
  }
  return fallback || `sd-cli failed (${code})`;
}

function isBackendInitLine(line) {
  return /^(ggml_cuda_init|ggml_vulkan|load_backend):/i.test(line);
}

/**
 * @param {string} modelPath
 * @param {string} prompt
 * @param {string} outputPath
 * @param {{ signal?: AbortSignal, modelRoot?: string, modelsCacheDir?: string, initImage?: string | null, referenceImage?: string | null }} [opts]
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

  const attached = attachedImage(opts);
  const imageSize = attached ? readImageSize(attached.path) : null;
  const result = await runOnce(
    buildSdCliArgs({
      modelPath,
      prompt: text,
      outputPath,
      components,
      initImage: opts.initImage,
      referenceImage: opts.referenceImage,
      width: imageSize ? imageSize.width : undefined,
      height: imageSize ? imageSize.height : undefined,
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
  fitInitCanvas,
  formatSdCliError,
  generateImage,
  imageSizeFromBuffer,
};
