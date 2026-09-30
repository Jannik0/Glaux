'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { rewriteNvFatbinSection, pruneCudaFatbinFile } = require('../scripts/cudaFatbin');

const FATBIN_MAGIC = 0xba55ed50;

/**
 * @param {number} kind
 * @param {number} arch
 * @param {Buffer} payload
 * @returns {Buffer}
 */
function fatbinImage(kind, arch, payload) {
  const header = Buffer.alloc(0x40);
  header.writeUInt16LE(kind, 0);
  header.writeUInt16LE(0x101, 2);
  header.writeUInt32LE(0x40, 4);
  header.writeBigUInt64LE(BigInt(payload.length), 8);
  header.writeUInt32LE(arch, 28);
  return Buffer.concat([header, payload]);
}

/**
 * @param {Buffer[]} images
 * @returns {Buffer}
 */
function fatbin(images) {
  const body = Buffer.concat(images);
  const header = Buffer.alloc(16);
  header.writeUInt32LE(FATBIN_MAGIC, 0);
  header.writeUInt16LE(1, 4);
  header.writeUInt16LE(16, 6);
  header.writeBigUInt64LE(BigInt(body.length), 8);
  return Buffer.concat([header, body]);
}

/**
 * @param {string} sectionName
 * @param {Buffer} payload
 * @returns {Buffer}
 */
function elfWithSection(sectionName, payload) {
  const shstr = Buffer.concat([
    Buffer.from('\0'),
    Buffer.from('.shstrtab\0'),
    Buffer.from(`${sectionName}\0`),
  ]);
  const shstrNameOff = 1;
  const sectionNameOff = 1 + Buffer.byteLength('.shstrtab\0');
  const shentsize = 64;
  const shnum = 3;
  const payloadOff = 64;
  const shstrOff = payloadOff + payload.length;
  const shoff = shstrOff + shstr.length;
  const buf = Buffer.alloc(shoff + shnum * shentsize);
  buf[0] = 0x7f;
  buf[1] = 0x45;
  buf[2] = 0x4c;
  buf[3] = 0x46;
  buf[4] = 2;
  buf[5] = 1;
  buf[6] = 1;
  buf.writeUInt16LE(3, 16);
  buf.writeUInt16LE(0x3e, 18);
  buf.writeUInt32LE(1, 20);
  buf.writeBigUInt64LE(BigInt(shoff), 40);
  buf.writeUInt16LE(64, 52);
  buf.writeUInt16LE(shentsize, 58);
  buf.writeUInt16LE(shnum, 60);
  buf.writeUInt16LE(2, 62);
  payload.copy(buf, payloadOff);
  shstr.copy(buf, shstrOff);

  const writeShdr = (index, nameOff, offset, size) => {
    const at = shoff + index * shentsize;
    buf.writeUInt32LE(nameOff, at);
    buf.writeUInt32LE(1, at + 4);
    buf.writeBigUInt64LE(BigInt(offset), at + 24);
    buf.writeBigUInt64LE(BigInt(size), at + 32);
  };
  writeShdr(1, sectionNameOff, payloadOff, payload.length);
  writeShdr(2, shstrNameOff, shstrOff, shstr.length);
  return buf;
}

describe('rewriteNvFatbinSection', () => {
  it('drops cubins outside the list and keeps 12.x PTX', () => {
    const section = fatbin([
      fatbinImage(2, 86, Buffer.from('CUBIN86')),
      fatbinImage(2, 121, Buffer.from('CUBIN121-EXTRA')),
      fatbinImage(1, 121, Buffer.from('PTX121')),
      fatbinImage(2, 103, Buffer.from('CUBIN103')),
    ]);
    const rewritten = rewriteNvFatbinSection(section);
    assert.ok(rewritten);
    assert.equal(rewritten.droppedImages, 2);
    assert.equal(rewritten.bytes.length, section.length);
    assert.equal(rewritten.bytes.includes(Buffer.from('CUBIN86')), true);
    assert.equal(rewritten.bytes.includes(Buffer.from('PTX121')), true);
    assert.equal(rewritten.bytes.includes(Buffer.from('CUBIN121-EXTRA')), false);
    assert.equal(rewritten.bytes.includes(Buffer.from('CUBIN103')), false);
    const again = rewriteNvFatbinSection(rewritten.bytes);
    assert.ok(again);
    assert.equal(again.droppedImages, 0);
  });

  it('keeps later fatbins at their original offsets', () => {
    const first = fatbin([
      fatbinImage(2, 86, Buffer.from('KEEP-A')),
      fatbinImage(2, 121, Buffer.from('DROP-A')),
    ]);
    const second = fatbin([fatbinImage(2, 80, Buffer.from('KEEP-B'))]);
    const section = Buffer.concat([first, second]);
    const secondAt = first.length;
    const rewritten = rewriteNvFatbinSection(section);
    assert.ok(rewritten);
    assert.equal(rewritten.bytes.readUInt32LE(secondAt), FATBIN_MAGIC);
    assert.equal(rewritten.bytes.includes(Buffer.from('KEEP-B')), true);
    assert.equal(rewritten.bytes.includes(Buffer.from('DROP-A')), false);
    assert.equal(rewritten.bytes.subarray(secondAt).equals(second), true);
  });

  it('leaves a section alone when it does not parse', () => {
    const broken = Buffer.from('not a fatbin section!!!!');
    assert.equal(rewriteNvFatbinSection(broken), null);
  });
});

describe('pruneCudaFatbinFile', () => {
  it('rewrites .nv_fatbin inside an ELF and skips symlinks', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'glaux-fatbin-'));
    const file = path.join(dir, 'libdemo.so');
    const payload = fatbin([
      fatbinImage(2, 75, Buffer.from('KEEP75')),
      fatbinImage(2, 70, Buffer.from('DROP70')),
    ]);
    fs.writeFileSync(file, elfWithSection('.nv_fatbin', payload));
    const before = fs.statSync(file).size;
    const result = pruneCudaFatbinFile(file);
    assert.ok(result);
    assert.ok(result.droppedBytes > 0);
    assert.equal(fs.statSync(file).size, before);
    const after = fs.readFileSync(file);
    assert.equal(after.includes(Buffer.from('KEEP75')), true);
    assert.equal(after.includes(Buffer.from('DROP70')), false);

    const link = path.join(dir, 'libdemo.so.1');
    fs.symlinkSync('libdemo.so', link);
    assert.equal(pruneCudaFatbinFile(link), null);
    fs.rmSync(dir, { recursive: true, force: true });
  });
});
