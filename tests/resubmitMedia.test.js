'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { buildInferenceOpenAIMessages } = require('../engines/llamacpp/chat');
const { pickPython } = require('../engines/common/runtimePaths');

function imagePart(file) {
  return { type: 'image', path: file };
}

describe('resubmit strips earlier assistant images', () => {
  it('keeps assistant images only on the last message unless resubmit is on', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-resubmit-'));
    const file = path.join(root, 'shot.png');
    fs.writeFileSync(file, 'png');
    try {
      const history = [
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'first' },
            imagePart(file),
          ],
        },
        { role: 'user', content: [{ type: 'text', text: 'next' }] },
      ];
      const off = buildInferenceOpenAIMessages(history, false);
      assert.equal(off[0].content, 'first');
      assert.equal(off[1].content, 'next');
      const on = buildInferenceOpenAIMessages(history, true);
      assert.equal(on[0].content.some((part) => part.type === 'image_url'), true);

      const lastAssistant = [
        { role: 'user', content: [{ type: 'text', text: 'draw' }, imagePart(file)] },
        {
          role: 'assistant',
          content: [{ type: 'text', text: 'done' }, imagePart(file)],
        },
      ];
      const kept = buildInferenceOpenAIMessages(lastAssistant, false);
      assert.equal(kept[0].content, 'draw');
      assert.equal(kept[1].content.some((part) => part.type === 'image_url'), true);
      const sent = buildInferenceOpenAIMessages(lastAssistant, true);
      assert.equal(sent[0].content.some((part) => part.type === 'image_url'), true);
      assert.equal(sent[1].content.some((part) => part.type === 'image_url'), true);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});

const PYTHON = pickPython();

function pythonSkipReason(command) {
  const result = spawnSync(command, ['-c', 'import sys'], {
    encoding: 'utf8',
    windowsHide: true,
    timeout: 15000,
  });
  if (result.status === 0) {
    return false;
  }
  const output = `${result.stderr || ''}\n${result.stdout || ''}`;
  const notFound =
    result.error?.code === 'ENOENT' ||
    result.status === 9009 ||
    /Python was not found/i.test(output);
  if (notFound) {
    return `No Python interpreter found (${command}).`;
  }
  const detail = (result.stderr || result.stdout || result.error?.message || `exit ${result.status}`).trim();
  return `Python interpreter failed to start (${command}): ${detail}`;
}

const PYTHON_SKIP = pythonSkipReason(PYTHON);

const SCRIPT = String.raw`
import sys
import types
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
hf = repo / "engines" / "huggingface"
sys.path.insert(0, str(hf))

worker = types.ModuleType("worker")
worker.__path__ = [str(hf / "worker")]
worker.__package__ = "worker"
sys.modules["worker"] = worker

import worker.context as context

history = [
    {"role": "assistant", "content": [
        {"type": "text", "text": "first"},
        {"type": "image", "path": "/tmp/shot.png"},
    ]},
    {"role": "user", "content": [{"type": "text", "text": "next"}]},
]
off = context._context_for_inference(history, resubmit=False)
assert off[0]["content"] == [{"type": "text", "text": "first"}]
assert off[1]["content"] == [{"type": "text", "text": "next"}]
on = context._context_for_inference(history, resubmit=True)
assert any(part.get("type") == "image" for part in on[0]["content"])

last = [
    {"role": "user", "content": [
        {"type": "text", "text": "draw"},
        {"type": "image", "path": "/tmp/shot.png"},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "done"},
        {"type": "image", "path": "/tmp/shot.png"},
    ]},
]
kept = context._context_for_inference(last, resubmit=False)
assert kept[0]["content"] == [{"type": "text", "text": "draw"}]
assert any(part.get("type") == "image" for part in kept[1]["content"])
sent = context._context_for_inference(last, resubmit=True)
assert any(part.get("type") == "image" for part in sent[0]["content"])
assert any(part.get("type") == "image" for part in sent[1]["content"])
`;

describe('huggingface resubmit strips earlier assistant images', () => {
  it('matches the llama.cpp rule for assistant image parts', { skip: PYTHON_SKIP }, () => {
    const result = spawnSync(PYTHON, ['-', path.resolve(__dirname, '..')], {
      input: SCRIPT,
      encoding: 'utf8',
      windowsHide: true,
    });
    assert.equal(
      result.status,
      0,
      result.stderr || result.stdout || result.error?.message || `${PYTHON} failed`
    );
  });
});
