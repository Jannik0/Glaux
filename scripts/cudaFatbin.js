'use strict';

/**
 * Drop cubin images outside the ggml architecture list from prebuilt CUDA
 * libraries. nvprune rejects linked .so/.dll files, so this rewrites the
 * `.nv_fatbin` section in place and zeroes the reclaimed bytes. Each fatbin
 * stays at its original offset: `.nvFatBinSegment` wrappers point at those
 * offsets, so later fatbins must not move. Virtual addresses stay put;
 * the zeros disappear in the compressed installer.
 *
 * Fail closed: a section that does not parse is left untouched.
 */

const fs = require('fs');
const path = require('path');
const { keepCudaFatbinImage } = require('../engines/common/cudaArch');

const FATBIN_MAGIC = 0xba55ed50;
const ELF_MAGIC = 0x464c457f;
const PE_MAGIC = 0x00004550;
const DOS_MAGIC = 0x5a4d;

/**
 * @param {Buffer} section
 * @returns {{ bytes: Buffer, droppedBytes: number, keptImages: number, droppedImages: number } | null}
 */
function rewriteNvFatbinSection(section) {
  const src = Buffer.from(section);
  const out = Buffer.from(section);
  let pos = 0;
  let droppedBytes = 0;
  let droppedImages = 0;
  let keptImages = 0;

  while (pos < src.length) {
    if (pos + 16 > src.length) {
      if (!isZero(src, pos, src.length)) {
        return null;
      }
      break;
    }
    if (src.readUInt32LE(pos) !== FATBIN_MAGIC) {
      if (src[pos] !== 0) {
        return null;
      }
      pos += 1;
      continue;
    }

    const headerSize = src.readUInt16LE(pos + 6);
    const fatSize = Number(src.readBigUInt64LE(pos + 8));
    if (headerSize < 16 || headerSize > 1024 || !Number.isSafeInteger(fatSize) || fatSize < 0) {
      return null;
    }
    if (pos + headerSize + fatSize > src.length) {
      return null;
    }

    const body = src.subarray(pos + headerSize, pos + headerSize + fatSize);
    /** @type {Buffer[]} */
    const kept = [];
    let entry = 0;
    let droppedHere = 0;
    while (entry < body.length) {
      if (entry + 16 > body.length) {
        if (!isZero(body, entry, body.length)) {
          return null;
        }
        break;
      }
      const kind = body.readUInt16LE(entry);
      const header = body.readUInt32LE(entry + 4);
      const code = Number(body.readBigUInt64LE(entry + 8));
      if (
        kind === 0 ||
        header < 16 ||
        header > 0x1000 ||
        !Number.isSafeInteger(code) ||
        code < 0 ||
        entry + header + code > body.length
      ) {
        return null;
      }
      const arch = header >= 32 ? body.readUInt32LE(entry + 28) : null;
      const span = header + code;
      if (arch == null || keepCudaFatbinImage(kind, arch)) {
        kept.push(Buffer.from(body.subarray(entry, entry + span)));
        keptImages += 1;
      } else {
        droppedHere += 1;
        droppedImages += 1;
        droppedBytes += span;
      }
      entry += span;
    }

    if (droppedHere) {
      const newBody = kept.length ? Buffer.concat(kept) : Buffer.alloc(0);
      if (newBody.length > fatSize) {
        return null;
      }
      out.writeBigUInt64LE(BigInt(newBody.length), pos + 8);
      if (newBody.length) {
        newBody.copy(out, pos + headerSize);
      }
      const newEnd = pos + headerSize + newBody.length;
      const oldEnd = pos + headerSize + fatSize;
      out.fill(0, newEnd, oldEnd);
    }
    // Next fatbin stays at its original offset. Wrappers in .nvFatBinSegment
    // point at these offsets.
    pos += headerSize + fatSize;
  }

  if (droppedImages === 0) {
    return { bytes: src, droppedBytes: 0, keptImages, droppedImages: 0 };
  }
  return { bytes: out, droppedBytes, keptImages, droppedImages };
}

/**
 * @param {Buffer} buf
 * @param {number} start
 * @param {number} end
 * @returns {boolean}
 */
function isZero(buf, start, end) {
  for (let i = start; i < end; i += 1) {
    if (buf[i] !== 0) {
      return false;
    }
  }
  return true;
}

/**
 * @param {Buffer} buf
 * @returns {{ name: string, offset: number, size: number }[]}
 */
function elfNvFatbinSections(buf) {
  if (buf.length < 64 || buf.readUInt32LE(0) !== ELF_MAGIC || buf[4] !== 2 || buf[5] !== 1) {
    return [];
  }
  const shoff = Number(buf.readBigUInt64LE(40));
  const shentsize = buf.readUInt16LE(58);
  const shnum = buf.readUInt16LE(60);
  const shstrndx = buf.readUInt16LE(62);
  if (shentsize < 64 || shnum === 0 || shstrndx >= shnum) {
    return [];
  }
  if (!Number.isSafeInteger(shoff) || shoff < 0 || shoff + shnum * shentsize > buf.length) {
    return [];
  }

  const strHeader = shoff + shstrndx * shentsize;
  const strOff = Number(buf.readBigUInt64LE(strHeader + 24));
  const strSize = Number(buf.readBigUInt64LE(strHeader + 32));
  if (!Number.isSafeInteger(strOff) || !Number.isSafeInteger(strSize) || strOff + strSize > buf.length) {
    return [];
  }

  /** @type {{ name: string, offset: number, size: number }[]} */
  const sections = [];
  for (let i = 0; i < shnum; i += 1) {
    const header = shoff + i * shentsize;
    const nameOff = buf.readUInt32LE(header);
    if (strOff + nameOff >= buf.length) {
      continue;
    }
    const nameEnd = buf.indexOf(0, strOff + nameOff);
    const name = buf.toString('latin1', strOff + nameOff, nameEnd === -1 ? buf.length : nameEnd);
    if (name !== '.nv_fatbin') {
      continue;
    }
    const offset = Number(buf.readBigUInt64LE(header + 24));
    const size = Number(buf.readBigUInt64LE(header + 32));
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(size) || offset < 0 || size < 0) {
      continue;
    }
    if (offset + size > buf.length) {
      continue;
    }
    sections.push({ name, offset, size });
  }
  return sections;
}

/**
 * @param {Buffer} buf
 * @returns {{ name: string, offset: number, size: number }[]}
 */
function peNvFatbinSections(buf) {
  if (buf.length < 0x40 || buf.readUInt16LE(0) !== DOS_MAGIC) {
    return [];
  }
  const lfanew = buf.readUInt32LE(0x3c);
  if (lfanew <= 0 || lfanew + 24 > buf.length || buf.readUInt32LE(lfanew) !== PE_MAGIC) {
    return [];
  }
  const coff = lfanew + 4;
  const sectionCount = buf.readUInt16LE(coff + 2);
  const symbolPointer = buf.readUInt32LE(coff + 8);
  const symbolCount = buf.readUInt32LE(coff + 12);
  const optionalSize = buf.readUInt16LE(coff + 16);
  const sectionTable = coff + 20 + optionalSize;
  if (sectionTable + sectionCount * 40 > buf.length) {
    return [];
  }
  const stringTable = symbolPointer + symbolCount * 18;

  /** @type {{ name: string, offset: number, size: number }[]} */
  const sections = [];
  for (let i = 0; i < sectionCount; i += 1) {
    const entry = sectionTable + i * 40;
    let name = buf.toString('latin1', entry, entry + 8).replace(/\0+$/, '');
    if (name.startsWith('/')) {
      const strOff = Number(name.slice(1));
      const at = stringTable + strOff;
      if (Number.isInteger(strOff) && at >= 0 && at < buf.length) {
        const end = buf.indexOf(0, at);
        if (end > at) {
          name = buf.toString('latin1', at, end);
        }
      }
    }
    if (name !== '.nv_fatbin' && name !== '.nv_fatb') {
      continue;
    }
    const rawSize = buf.readUInt32LE(entry + 16);
    const rawPtr = buf.readUInt32LE(entry + 20);
    if (rawPtr + rawSize > buf.length) {
      continue;
    }
    sections.push({ name, offset: rawPtr, size: rawSize });
  }
  return sections;
}

/**
 * @param {string} filePath
 * @returns {{ droppedBytes: number, droppedImages: number } | null}
 */
function pruneCudaFatbinFile(filePath) {
  let stat;
  try {
    stat = fs.lstatSync(filePath);
  } catch {
    return null;
  }
  if (!stat.isFile() || stat.isSymbolicLink()) {
    return null;
  }

  const buf = fs.readFileSync(filePath);
  const sections =
    buf.length >= 4 && buf.readUInt32LE(0) === ELF_MAGIC
      ? elfNvFatbinSections(buf)
      : buf.length >= 2 && buf.readUInt16LE(0) === DOS_MAGIC
        ? peNvFatbinSections(buf)
        : [];
  if (!sections.length) {
    return null;
  }

  let droppedBytes = 0;
  let droppedImages = 0;
  let changed = false;
  for (const section of sections) {
    if (section.size === 0) {
      continue;
    }
    const rewritten = rewriteNvFatbinSection(buf.subarray(section.offset, section.offset + section.size));
    if (!rewritten || rewritten.droppedImages === 0) {
      continue;
    }
    rewritten.bytes.copy(buf, section.offset);
    droppedBytes += rewritten.droppedBytes;
    droppedImages += rewritten.droppedImages;
    changed = true;
  }
  if (!changed) {
    return null;
  }

  const tmp = `${filePath}.${process.pid}.fatbin-tmp`;
  fs.writeFileSync(tmp, buf);
  try {
    fs.chmodSync(tmp, stat.mode);
  } catch {
    /* mode is best-effort */
  }
  fs.renameSync(tmp, filePath);
  return { droppedBytes, droppedImages };
}

/**
 * @param {string} name
 * @returns {boolean}
 */
function isCudaLibraryFileName(name) {
  return /\.(?:so|dll)(?:\.\d+)*$/i.test(name);
}

/**
 * @param {string} root
 * @returns {{ file: string, droppedBytes: number, droppedImages: number }[]}
 */
function pruneCudaFatbinsInTree(root) {
  /** @type {{ file: string, droppedBytes: number, droppedImages: number }[]} */
  const changed = [];
  walk(root, changed);
  if (changed.length) {
    const bytes = changed.reduce((sum, item) => sum + item.droppedBytes, 0);
    const names = changed.map((item) => path.basename(item.file)).join(', ');
    console.log(
      `Pruned CUDA fatbins in ${changed.length} file(s), zeroed ${(bytes / (1024 * 1024)).toFixed(1)} MiB ` +
        `outside the bundled architectures: ${names}`
    );
  }
  return changed;
}

/**
 * @param {string} dir
 * @param {{ file: string, droppedBytes: number, droppedImages: number }[]} changed
 */
function walk(dir, changed) {
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.isSymbolicLink()) {
      continue;
    }
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full, changed);
      continue;
    }
    if (!entry.isFile() || !isCudaLibraryFileName(entry.name)) {
      continue;
    }
    const result = pruneCudaFatbinFile(full);
    if (result) {
      changed.push({ file: full, ...result });
    }
  }
}

module.exports = {
  rewriteNvFatbinSection,
  pruneCudaFatbinFile,
  pruneCudaFatbinsInTree,
  isCudaLibraryFileName,
};

if (require.main === module) {
  const roots = process.argv.slice(2);
  if (!roots.length) {
    console.error('Usage: node scripts/cudaFatbin.js <dir>...');
    process.exit(1);
  }
  for (const root of roots) {
    pruneCudaFatbinsInTree(path.resolve(root));
  }
}
