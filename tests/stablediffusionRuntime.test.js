'use strict';

const { describe, it, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  STAGED_LICENSE_FILES,
  checkStableDiffusionRuntime,
} = require('../scripts/check-stablediffusion-runtime');

describe('stable-diffusion runtime license check', () => {
  let root;

  after(() => {
    if (root) {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it('names each missing license file and accepts them once they are staged', () => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-licenses-'));
    const missing = checkStableDiffusionRuntime({ vendor: root, cudaDir: path.join(root, 'cuda') });
    const text = missing.join('\n');
    for (const name of STAGED_LICENSE_FILES) {
      assert.match(text, new RegExp(name.replace('.', '\\.')));
    }
    assert.match(text, /must sit next to sd-cli/);

    for (const name of STAGED_LICENSE_FILES) {
      fs.writeFileSync(path.join(root, name), 'notice');
    }
    const remaining = checkStableDiffusionRuntime({ vendor: root, cudaDir: path.join(root, 'cuda') }).join('\n');
    assert.equal(/Missing third-party license notice/.test(remaining), false);
  });
});
