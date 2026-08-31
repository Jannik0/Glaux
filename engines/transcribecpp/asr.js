'use strict';

/**
 * ASR via bundled transcribe-cli: ensure 16 kHz mono WAV, choose offline vs
 * cache-aware streaming args, parse progressive stdout into onToken deltas.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const fsp = require('fs').promises;
const os = require('os');
const path = require('path');
const { pickFfmpeg } = require('../common/ffmpeg');
const { isForceCpu } = require('../common/gpuRuntime');
const { runTranscribeCli } = require('./cli');

const STREAM_CHUNK_MS = 1120;
const STREAM_ATT_RIGHT = 13;

/** Heuristics for models that support cache-aware / chunked streaming in the CLI. */
const STREAMING_MODEL_RE =
  /nemotron|streaming|moonshine-streaming|voxtral.*realtime|multitalker-parakeet/i;

/**
 * @param {string} modelId
 * @param {string} modelPath
 * @returns {boolean}
 */
function usesCacheAwareStreaming(modelId, modelPath) {
  const haystack = `${modelId || ''} ${path.basename(modelPath || '')}`;
  return STREAMING_MODEL_RE.test(haystack);
}

/**
 * @param {string} _modelId
 * @param {string} _modelPath
 * @returns {string | null} CLI --language value, or null to omit
 *
 * Always omit `--language` so the model auto-detects (empty hint →
 * prompt.auto_id / family default). Do not pass the literal string "auto":
 * that fails caps.languages validation with "unsupported language".
 */
function languageFlagForModel(_modelId, _modelPath) {
  return null;
}

/**
 * @param {{
 *   modelPath: string,
 *   wavPath: string,
 *   streaming?: boolean,
 *   language?: string | null,
 *   forceCpu?: boolean,
 * }} opts
 * @returns {string[]}
 */
function buildTranscribeCliArgs(opts) {
  const forceCpu = opts.forceCpu != null ? Boolean(opts.forceCpu) : isForceCpu();
  const args = ['-m', opts.modelPath, '--backend', forceCpu ? 'cpu' : 'auto'];
  if (opts.language) {
    args.push('--language', opts.language);
  }
  if (opts.streaming) {
    args.push(
      '--stream-chunk-ms',
      String(STREAM_CHUNK_MS),
      '--stream-att-right',
      String(STREAM_ATT_RIGHT)
    );
  }
  args.push(opts.wavPath);
  return args;
}

/**
 * Probe whether path is already 16 kHz mono WAV (rough: extension + ffprobe if available).
 * Always re-encode through ffmpeg when available so CLI sample-rate checks pass.
 *
 * @param {string} audioPath
 * @param {{ signal?: AbortSignal }} [opts]
 * @returns {Promise<string>} Path to a 16 kHz mono WAV (may be a temp file)
 */
async function ensureSixteenKhzMonoWav(audioPath, opts = {}) {
  const abs = path.resolve(audioPath);
  await fsp.access(abs);

  const ffmpeg = pickFfmpeg();
  if (!ffmpeg) {
    throw new Error(
      'ffmpeg not found (needed to convert audio to 16 kHz mono WAV). ' +
        'Run npm run build:ffmpeg.'
    );
  }

  const tmpDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'glaux-asr-'));
  const outPath = path.join(tmpDir, `${path.basename(abs, path.extname(abs))}-16k.wav`);

  await new Promise((resolve, reject) => {
    const args = [
      '-y',
      '-i',
      abs,
      '-ar',
      '16000',
      '-ac',
      '1',
      '-c:a',
      'pcm_s16le',
      outPath,
    ];
    const child = spawn(ffmpeg, args, { windowsHide: true });
    let err = '';
    const onAbort = () => {
      try {
        child.kill();
      } catch {
        /* ignore */
      }
    };
    if (opts.signal) {
      if (opts.signal.aborted) {
        onAbort();
      } else {
        opts.signal.addEventListener('abort', onAbort, { once: true });
      }
    }
    child.stderr.on('data', (d) => {
      err += d.toString('utf8');
    });
    child.on('error', (e) => {
      if (opts.signal) {
        opts.signal.removeEventListener('abort', onAbort);
      }
      reject(e);
    });
    child.on('close', (code) => {
      if (opts.signal) {
        opts.signal.removeEventListener('abort', onAbort);
      }
      if (code === 0 && fs.existsSync(outPath)) {
        resolve();
      } else if (opts.signal && opts.signal.aborted) {
        reject(new Error('Aborted'));
      } else {
        reject(new Error(err.trim() || `ffmpeg failed (${code})`));
      }
    });
  });

  return outPath;
}

/**
 * Extract a progressive partial string from a CLI stdout line.
 * Streaming feed lines look like: feed[ 0]: ... partial="hello world"
 *
 * @param {string} line
 * @returns {string | null}
 */
function extractPartialFromLine(line) {
  const m = line.match(/partial="((?:\\.|[^"\\])*)"/);
  if (!m) {
    return null;
  }
  return m[1]
    .replace(/\\n/g, '\n')
    .replace(/\\r/g, '\r')
    .replace(/\\t/g, '\t')
    .replace(/\\"/g, '"')
    .replace(/\\\\/g, '\\');
}

/**
 * Final offline / post-stream line: `text: ...`
 * @param {string} line
 * @returns {string | null}
 */
function extractFinalTextFromLine(line) {
  const m = line.match(/^text:\s*(.*)$/);
  if (!m) {
    return null;
  }
  const text = m[1].trim();
  if (text === '(empty)') {
    return '';
  }
  return text;
}

/**
 * @param {string} modelPath
 * @param {string} modelId
 * @param {string} audioPath
 * @param {{ onToken?: (chunk: string) => void, onReplace?: (text: string) => void, signal?: AbortSignal }} [opts]
 * @returns {Promise<string>}
 */
async function transcribeAudio(modelPath, modelId, audioPath, opts = {}) {
  let wavPath = audioPath;
  let tempWav = false;
  try {
    const converted = await ensureSixteenKhzMonoWav(audioPath, { signal: opts.signal });
    if (converted !== path.resolve(audioPath)) {
      wavPath = converted;
      tempWav = true;
    }

    const streaming = usesCacheAwareStreaming(modelId, modelPath);
    const args = buildTranscribeCliArgs({
      modelPath,
      wavPath,
      streaming,
      language: languageFlagForModel(modelId, modelPath),
    });

    let emitted = '';
    let finalText = '';
    const useReplace = typeof opts.onReplace === 'function';

    const publish = (text) => {
      if (typeof text !== 'string') {
        return;
      }
      if (useReplace) {
        if (text !== emitted) {
          opts.onReplace(text);
          emitted = text;
        }
        return;
      }
      // Append-only fallback when the UI has no snapshot path.
      if (text.startsWith(emitted)) {
        const delta = text.slice(emitted.length);
        if (delta && opts.onToken) {
          opts.onToken(delta);
        }
        emitted = text;
      }
    };

    const result = await runTranscribeCli(args, {
      signal: opts.signal,
      onStdoutLine: (line) => {
        const partial = extractPartialFromLine(line);
        if (partial != null) {
          publish(partial);
          return;
        }
        const fin = extractFinalTextFromLine(line);
        if (fin != null) {
          finalText = fin;
        }
      },
    });

    if (opts.signal && opts.signal.aborted) {
      return emitted || finalText || '';
    }

    if (result.code !== 0) {
      const errText = (result.stderr || result.stdout || '').replace(/\s+/g, ' ').trim();
      throw new Error(
        errText.length > 400
          ? `${errText.slice(0, 400)}…`
          : errText || `transcribe-cli failed (${result.code})`
      );
    }

    const reply = finalText || emitted;
    if (reply) {
      publish(reply);
    }

    return reply;
  } finally {
    if (tempWav && wavPath) {
      try {
        await fsp.rm(path.dirname(wavPath), { recursive: true, force: true });
      } catch {
        /* ignore */
      }
    }
  }
}

module.exports = {
  usesCacheAwareStreaming,
  languageFlagForModel,
  ensureSixteenKhzMonoWav,
  extractPartialFromLine,
  extractFinalTextFromLine,
  transcribeAudio,
  pickFfmpeg,
  buildTranscribeCliArgs,
};
