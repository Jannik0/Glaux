'use strict';

const { describe, it, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  patchStaticLibwebp,
  stageThirdPartyLicenses,
} = require('../scripts/build-stablediffusion');

const NOTICES = [
  ['thirdparty', 'libwebp', 'COPYING', 'LIBWEBP.COPYING'],
  ['thirdparty', 'libwebp', 'PATENTS', 'LIBWEBP.PATENTS'],
  ['thirdparty', 'oniguruma', 'COPYING', 'ONIGURUMA.COPYING'],
  ['thirdparty', 'LICENSE.darts_clone.txt', 'DARTS.LICENSE'],
  ['thirdparty', 'utf8proc', 'LICENSE.md', 'UTF8PROC.LICENSE.md'],
];

describe('stable-diffusion build license helpers', () => {
  let root;

  after(() => {
    if (root) {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  function srcWithNeedle() {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-build-'));
    const cmake = path.join(root, 'thirdparty');
    fs.mkdirSync(cmake, { recursive: true });
    const file = path.join(cmake, 'CMakeLists.txt');
    fs.writeFileSync(file, 'add_subdirectory(libwebp EXCLUDE_FROM_ALL)\n');
    return file;
  }

  it('patches libwebp to link statically and does not apply twice', () => {
    const file = srcWithNeedle();
    patchStaticLibwebp(root);
    const once = fs.readFileSync(file, 'utf8');
    assert.match(once, /GLAUX_STATIC_LIBWEBP/);
    assert.match(once, /set\(BUILD_SHARED_LIBS OFF\)/);
    assert.equal(once.split('GLAUX_STATIC_LIBWEBP').length, 2);
    patchStaticLibwebp(root);
    assert.equal(fs.readFileSync(file, 'utf8'), once);
  });

  it('throws when the libwebp subdirectory line is missing', () => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-build-'));
    const cmake = path.join(root, 'thirdparty');
    fs.mkdirSync(cmake, { recursive: true });
    fs.writeFileSync(path.join(cmake, 'CMakeLists.txt'), 'add_subdirectory(other)\n');
    assert.throws(() => patchStaticLibwebp(root), /Could not locate the libwebp subdirectory/);
  });

  it('copies the five license files and throws when one source is missing', () => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-sd-build-'));
    const out = path.join(root, 'out');
    fs.mkdirSync(out);
    for (const parts of NOTICES) {
      const rel = parts.slice(0, -1);
      const file = path.join(root, ...rel);
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, parts[parts.length - 1]);
    }
    stageThirdPartyLicenses(root, out);
    for (const parts of NOTICES) {
      const dest = parts[parts.length - 1];
      assert.equal(fs.readFileSync(path.join(out, dest), 'utf8'), dest);
    }
    fs.rmSync(path.join(root, 'thirdparty', 'libwebp', 'PATENTS'));
    assert.throws(() => stageThirdPartyLicenses(root, out), /Missing third-party license text/);
  });
});
