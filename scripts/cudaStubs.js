'use strict';

/**
 * Replace NVIDIA libraries that libtorch links but Glaux never calls with
 * tiny loader stubs, on Windows and Linux. The real binaries stay out of the
 * installer; the SONAME / DLL name stays so CPython's RTLD_NOW import of
 * torch succeeds.
 *
 * Stubbed: NCCL (multi-GPU collectives), cuSPARSELt (2:4 structured sparsity),
 * NVSHMEM (multi-GPU shared memory), and cuFile (GPUDirect Storage).
 * cuDNN, cuFFT, cuRAND, NVRTC, cuSOLVER, cuSPARSE, and the shared cublas
 * runtime are not stubbed.
 */

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { elfNeeded, findSitePackages, isNativeLibName, which } = require('./gpuBackends');

/**
 * NVIDIA redistributable file names that are swapped for stubs or deleted
 * once the stub is in place. Torch's own libtorch_nvshmem is not included.
 *
 * @param {string} name basename or path
 * @returns {boolean}
 */
function isStubbedCudaFamilyFile(name) {
  const base = path.basename(String(name));
  if (/^libtorch_/i.test(base) || /\.(py|pyi|pyc|pyo|pth)$/i.test(base)) {
    return false;
  }
  return (
    /^(lib)?nccl(\.|$)/i.test(base) ||
    /^(lib)?cusparseLt/i.test(base) ||
    /^(lib)?nvshmem/i.test(base) ||
    /^nvshmem_/i.test(base) ||
    /^(lib)?cufile/i.test(base)
  );
}

/**
 * @param {string} dir
 * @param {string[]} out
 */
function collectFiles(dir, out) {
  let ents;
  try {
    ents = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const ent of ents) {
    if (ent.name === '__pycache__' || ent.name === '.git') {
      continue;
    }
    const full = path.join(dir, ent.name);
    if (ent.isDirectory() && !ent.isSymbolicLink()) {
      collectFiles(full, out);
      continue;
    }
    out.push(full);
  }
}

/**
 * @param {Buffer} buf
 * @param {number} rva
 * @param {{ va: number, rawSize: number, rawPtr: number }[]} sections
 * @returns {number | null}
 */
function rvaToOffset(buf, rva, sections) {
  for (const section of sections) {
    const span = Math.max(section.rawSize, 1);
    if (rva >= section.va && rva < section.va + span) {
      const off = section.rawPtr + (rva - section.va);
      if (off >= 0 && off < buf.length) {
        return off;
      }
    }
  }
  return null;
}

/**
 * @param {Buffer} buf
 * @param {number} off
 * @returns {string}
 */
function readCString(buf, off) {
  let end = off;
  while (end < buf.length && buf[end] !== 0) {
    end += 1;
  }
  return buf.toString('utf8', off, end);
}

/**
 * @param {Buffer} buf
 * @param {number} thunkRva
 * @param {{ va: number, rawSize: number, rawPtr: number }[]} sections
 * @param {boolean} pe32Plus
 * @returns {string[]}
 */
function readPeThunkNames(buf, thunkRva, sections, pe32Plus) {
  const names = [];
  if (!thunkRva) {
    return names;
  }
  let thunk = rvaToOffset(buf, thunkRva, sections);
  if (thunk == null) {
    return names;
  }
  const step = pe32Plus ? 8 : 4;
  for (let i = 0; i < 100000 && thunk + step <= buf.length; i += 1) {
    let ordinal = false;
    let nameRva = 0;
    if (pe32Plus) {
      const lo = buf.readUInt32LE(thunk);
      const hi = buf.readUInt32LE(thunk + 4);
      if (lo === 0 && hi === 0) {
        break;
      }
      ordinal = (hi & 0x80000000) !== 0;
      nameRva = lo;
    } else {
      const value = buf.readUInt32LE(thunk);
      if (value === 0) {
        break;
      }
      ordinal = (value & 0x80000000) !== 0;
      nameRva = value & 0x7fffffff;
    }
    if (!ordinal) {
      const nameOff = rvaToOffset(buf, nameRva, sections);
      if (nameOff != null && nameOff + 2 < buf.length) {
        const name = readCString(buf, nameOff + 2);
        if (name) {
          names.push(name);
        }
      }
    }
    thunk += step;
  }
  return names;
}

/**
 * DLL names and symbol names imported by a PE file, and names it exports.
 * @param {string} filePath
 * @returns {{ imports: string[], importSymbols: string[], exports: string[] } | null}
 */
function readPeImportsAndExports(filePath) {
  let buf;
  try {
    buf = fs.readFileSync(filePath);
  } catch {
    return null;
  }
  if (buf.length < 0x40 || buf[0] !== 0x4d || buf[1] !== 0x5a) {
    return null;
  }
  const peOff = buf.readUInt32LE(0x3c);
  if (peOff <= 0 || peOff + 24 > buf.length || buf.readUInt32LE(peOff) !== 0x4550) {
    return null;
  }
  const coff = peOff + 4;
  const nSections = buf.readUInt16LE(coff + 2);
  const optSize = buf.readUInt16LE(coff + 16);
  const opt = coff + 20;
  if (opt + optSize > buf.length) {
    return null;
  }
  const magic = buf.readUInt16LE(opt);
  const pe32Plus = magic === 0x20b;
  const dd = opt + (pe32Plus ? 112 : 96);
  if (dd + 16 > buf.length) {
    return null;
  }
  const exportRva = buf.readUInt32LE(dd);
  const importRva = buf.readUInt32LE(dd + 8);
  const sections = [];
  const sec = opt + optSize;
  for (let i = 0; i < nSections; i += 1) {
    const o = sec + i * 40;
    if (o + 24 > buf.length) {
      break;
    }
    sections.push({
      va: buf.readUInt32LE(o + 12),
      rawSize: buf.readUInt32LE(o + 16),
      rawPtr: buf.readUInt32LE(o + 20),
    });
  }

  const imports = [];
  const importSymbols = [];
  if (importRva) {
    let desc = rvaToOffset(buf, importRva, sections);
    if (desc != null) {
      for (let n = 0; n < 256 && desc + 20 <= buf.length; n += 1) {
        const nameRva = buf.readUInt32LE(desc + 12);
        const oft = buf.readUInt32LE(desc);
        const ft = buf.readUInt32LE(desc + 16);
        if (nameRva === 0 && oft === 0 && ft === 0) {
          break;
        }
        const nameOff = rvaToOffset(buf, nameRva, sections);
        if (nameOff != null) {
          imports.push(readCString(buf, nameOff));
        }
        importSymbols.push(...readPeThunkNames(buf, oft || ft, sections, pe32Plus));
        desc += 20;
      }
    }
  }

  const exports = [];
  if (exportRva) {
    const exp = rvaToOffset(buf, exportRva, sections);
    if (exp != null && exp + 40 <= buf.length) {
      const nNames = buf.readUInt32LE(exp + 24);
      const namesOff = rvaToOffset(buf, buf.readUInt32LE(exp + 32), sections);
      if (namesOff != null) {
        const count = Math.min(nNames, 100000);
        for (let i = 0; i < count; i += 1) {
          const pos = namesOff + i * 4;
          if (pos + 4 > buf.length) {
            break;
          }
          const noff = rvaToOffset(buf, buf.readUInt32LE(pos), sections);
          if (noff != null) {
            const name = readCString(buf, noff);
            if (name) {
              exports.push(name);
            }
          }
        }
      }
    }
  }
  return { imports, importSymbols, exports };
}

/**
 * @param {string} filePath
 * @returns {{ name: string, kind: 'func' | 'data', version: string | null }[]}
 */
function readDynamicExports(filePath) {
  if (process.platform === 'win32') {
    const pe = readPeImportsAndExports(filePath);
    if (!pe) {
      return [];
    }
    return pe.exports.map((name) => ({ name, kind: 'func', version: null }));
  }

  const nm = which('nm');
  if (!nm) {
    throw new Error('nm is required to build CUDA loader stubs. Install binutils and retry.');
  }
  const result = spawnSync(nm, ['-D', '--defined-only', filePath], {
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `nm failed for ${filePath}: ${(result.stderr || result.stdout || '').trim()}`
    );
  }
  const symbols = [];
  for (const line of String(result.stdout || '').split(/\r?\n/)) {
    const parts = line.trim().split(/\s+/);
    if (parts.length < 3) {
      continue;
    }
    const type = parts[parts.length - 2];
    const raw = parts[parts.length - 1];
    const versionSep = raw.indexOf('@');
    const name = versionSep === -1 ? raw : raw.slice(0, versionSep);
    let version = null;
    if (versionSep !== -1) {
      version = raw.slice(versionSep).replace(/^@+/, '');
    }
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
      continue;
    }
    if (version && !/^[A-Za-z0-9_.]+$/.test(version)) {
      version = null;
    }
    if (STUB_SYMBOL_DENY.has(name)) {
      continue;
    }
    if (type === 'T' || type === 'W') {
      symbols.push({ name, kind: 'func', version });
    } else if (type === 'D' || type === 'B' || type === 'R' || type === 'V') {
      symbols.push({ name, kind: 'data', version });
    }
  }
  return symbols;
}

const STUB_SYMBOL_DENY = new Set(['_init', '_fini', '_edata', '_end', '_bss_start', '__bss_start']);

/**
 * Dynamic symbols a kept library imports. Versions come from the definition
 * side when the stub is built; this list is only the names Torch asks for.
 * @param {string} filePath
 * @returns {string[]}
 */
function readUndefinedDynsymNames(filePath) {
  if (process.platform === 'win32') {
    const pe = readPeImportsAndExports(filePath);
    if (!pe) {
      return [];
    }
    return pe.importSymbols.filter((name) => name && !STUB_SYMBOL_DENY.has(name));
  }
  const nm = which('nm');
  if (!nm) {
    return [];
  }
  const result = spawnSync(nm, ['-D', '--undefined-only', filePath], {
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  });
  if (result.status !== 0) {
    return [];
  }
  const names = [];
  for (const line of String(result.stdout || '').split(/\r?\n/)) {
    const parts = line.trim().split(/\s+/);
    if (parts.length < 2) {
      continue;
    }
    const raw = parts[parts.length - 1];
    const name = raw.split('@')[0];
    if (name && !STUB_SYMBOL_DENY.has(name)) {
      names.push(name);
    }
  }
  return names;
}

/**
 * @param {string} filePath
 * @returns {string[]}
 */
function neededLibraryNames(filePath) {
  if (process.platform === 'win32') {
    const pe = readPeImportsAndExports(filePath);
    return pe ? pe.imports : [];
  }
  if (!isNativeLibName(path.basename(filePath))) {
    return [];
  }
  try {
    if (!fs.statSync(filePath).isFile() || fs.lstatSync(filePath).isSymbolicLink()) {
      return [];
    }
  } catch {
    return [];
  }
  return elfNeeded(filePath);
}

/**
 * @param {string} compiler
 * @param {string} cFile
 * @param {string} outFile
 * @param {string} soname
 * @param {string | null} mapFile
 */
function compileStub(compiler, cFile, outFile, soname, mapFile) {
  const base = path.basename(compiler).toLowerCase();
  const isCl = base === 'cl' || base === 'cl.exe';
  let args;
  if (isCl) {
    args = ['/nologo', '/LD', `/Fe:${outFile}`, cFile];
  } else if (process.platform === 'win32') {
    args = ['-shared', '-O2', '-o', outFile, cFile];
  } else {
    args = ['-shared', '-fPIC', '-O2', `-Wl,-soname,${soname}`];
    if (mapFile) {
      args.push(`-Wl,--version-script,${mapFile}`);
    }
    args.push('-o', outFile, cFile);
  }
  const result = spawnSync(compiler, args, { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
  if (result.status !== 0) {
    throw new Error(
      `Failed to compile CUDA loader stub ${soname}: ${(result.stderr || result.stdout || '').trim()}`
    );
  }
}

/**
 * @returns {string}
 */
function findStubCompiler() {
  if (process.platform === 'win32') {
    return which('cl') || which('gcc') || which('clang') || '';
  }
  return which('gcc') || which('cc') || which('clang') || '';
}

/**
 * @param {string} filePath
 * @param {{ name: string, kind: 'func' | 'data', version: string | null }[]} symbols
 * @param {string} compiler
 */
function writeStubOver(filePath, symbols, compiler) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-cuda-stub-'));
  try {
    const seen = new Set();
    const unique = [];
    for (const sym of symbols) {
      if (seen.has(sym.name)) {
        continue;
      }
      seen.add(sym.name);
      unique.push(sym);
    }
    if (!unique.length) {
      throw new Error(`No exported symbols to stub in ${filePath}`);
    }
    const cLines = [
      '#if defined(_WIN32)',
      '#define EXP __declspec(dllexport)',
      '#else',
      '#define EXP',
      '#endif',
    ];
    /** @type {Map<string, string[]>} */
    const byVersion = new Map();
    let plainCount = 0;
    for (const sym of unique) {
      if (sym.kind === 'data') {
        cLines.push(`EXP char ${sym.name}[8];`);
      } else {
        cLines.push(`EXP void ${sym.name}(void) {}`);
      }
      if (!sym.version) {
        plainCount += 1;
        continue;
      }
      const list = byVersion.get(sym.version) || [];
      list.push(sym.name);
      byVersion.set(sym.version, list);
    }
    const cFile = path.join(dir, 'stub.c');
    fs.writeFileSync(cFile, `${cLines.join('\n')}\n`);

    let mapFile = null;
    const versions = [...byVersion.keys()];
    if (versions.length && process.platform !== 'win32') {
      if (plainCount) {
        throw new Error(`Cannot stub ${path.basename(filePath)}: mixed versioned and unversioned exports`);
      }
      const chunks = versions.map((ver, index) => {
        const names = byVersion.get(ver) || [];
        const local = index === versions.length - 1 ? '\n  local:\n    *;' : '';
        return `${ver} {\n  global:\n    ${names.join(';\n    ')};${local}\n};`;
      });
      mapFile = path.join(dir, 'stub.map');
      fs.writeFileSync(mapFile, `${chunks.join('\n')}\n`);
    }

    const soname = path.basename(filePath);
    const outFile = path.join(dir, soname);
    compileStub(compiler, cFile, outFile, soname, mapFile);
    fs.rmSync(filePath, { force: true });
    fs.copyFileSync(outFile, filePath);
    if (process.platform !== 'win32') {
      fs.chmodSync(filePath, 0o755);
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

/**
 * @param {string} runtimeRoot
 * @returns {{ stubbed: string[], removedBytes: number }}
 */
function stubTorchUnusedCudaDeps(runtimeRoot) {
  const empty = { stubbed: [], removedBytes: 0 };
  if (process.platform === 'darwin' || !runtimeRoot || !fs.existsSync(runtimeRoot)) {
    return empty;
  }
  const site = findSitePackages(runtimeRoot);
  if (!site) {
    return empty;
  }

  const files = [];
  collectFiles(site, files);
  const imported = new Set();
  const needed = new Set();
  const remember = (name) => {
    if (!name) {
      return;
    }
    needed.add(process.platform === 'win32' ? name.toLowerCase() : name);
  };
  for (const file of files) {
    const base = path.basename(file);
    if (isStubbedCudaFamilyFile(base) || !isNativeLibName(base)) {
      continue;
    }
    for (const name of neededLibraryNames(file)) {
      remember(name);
    }
    for (const name of readUndefinedDynsymNames(file)) {
      imported.add(name);
    }
  }

  const targets = [];
  for (const file of files) {
    const base = path.basename(file);
    if (!isStubbedCudaFamilyFile(base) || !isNativeLibName(base)) {
      continue;
    }
    let st;
    try {
      st = fs.lstatSync(file);
    } catch {
      continue;
    }
    if (st.isSymbolicLink()) {
      continue;
    }
    const key = process.platform === 'win32' ? base.toLowerCase() : base;
    if (needed.has(key)) {
      targets.push(file);
    }
  }
  if (!targets.length) {
    return empty;
  }

  const compiler = findStubCompiler();
  if (!compiler) {
    throw new Error(
      'A C compiler (gcc, clang, or cl) is required to stub NCCL, cuSPARSELt, NVSHMEM, and cuFile.'
    );
  }

  let removedBytes = 0;
  const stubbed = [];
  const stubbedPaths = new Set();
  for (const file of targets) {
    const before = fs.statSync(file).size;
    const defined = readDynamicExports(file);
    const symbols = [];
    for (const sym of defined) {
      if (!imported.has(sym.name)) {
        continue;
      }
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(sym.name)) {
        throw new Error(
          `Cannot stub ${path.basename(file)}: imported symbol is not a C identifier (${sym.name})`
        );
      }
      symbols.push(sym);
    }
    if (!symbols.length) {
      throw new Error(`No imported symbols to stub in ${file}`);
    }
    writeStubOver(file, symbols, compiler);
    const after = fs.statSync(file).size;
    removedBytes += Math.max(0, before - after);
    stubbed.push(path.basename(file));
    stubbedPaths.add(path.resolve(file));
    console.log(
      `Stubbed ${path.basename(file)} (${(before / (1024 * 1024)).toFixed(1)} MB -> ${(after / (1024 * 1024)).toFixed(2)} MB)`
    );
  }

  const again = [];
  collectFiles(site, again);
  for (const file of again) {
    if (!isStubbedCudaFamilyFile(file) || stubbedPaths.has(path.resolve(file))) {
      continue;
    }
    let st;
    try {
      st = fs.lstatSync(file);
    } catch {
      continue;
    }
    if (st.isDirectory()) {
      continue;
    }
    if (st.isSymbolicLink()) {
      const targetName = path.basename(fs.readlinkSync(file));
      const key = process.platform === 'win32' ? targetName.toLowerCase() : targetName;
      if (needed.has(key)) {
        continue;
      }
    }
    removedBytes += st.size;
    fs.rmSync(file, { force: true });
  }

  console.log(
    `Replaced unused CUDA deps with loader stubs (${stubbed.join(', ')}); ` +
      `removed ${(removedBytes / (1024 * 1024)).toFixed(1)} MB`
  );
  return { stubbed, removedBytes };
}

module.exports = {
  isStubbedCudaFamilyFile,
  readPeImportsAndExports,
  stubTorchUnusedCudaDeps,
};
