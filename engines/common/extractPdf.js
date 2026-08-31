'use strict';

/**
 * Extract PDF text into a sibling `.md` file (### Page N sections).
 */

const fsp = require('fs').promises;
const path = require('path');
const pdfParse = require('pdf-parse');
const { PDF_EXTS, extOf } = require('./mediaKinds');

const EXTRACTED_TEXT_CONTROL_CHARS_RE = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g;

/**
 * Strip NUL/control characters and unpaired UTF-16 surrogates that PDF extractors
 * sometimes emit for malformed fonts/encodings (keeps tab/newline/CR intact).
 * @param {string} text
 * @returns {string}
 */
function sanitizeExtractedText(text) {
  if (!text) {
    return text;
  }
  const cleaned = String(text).replace(EXTRACTED_TEXT_CONTROL_CHARS_RE, '');
  // Drop unpaired UTF-16 surrogates (same intent as Python utf-16 encode/decode ignore).
  let out = '';
  for (let i = 0; i < cleaned.length; i++) {
    const c = cleaned.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      const next = cleaned.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        out += cleaned[i] + cleaned[i + 1];
        i += 1;
      }
    } else if (c < 0xdc00 || c > 0xdfff) {
      out += cleaned[i];
    }
  }
  return out;
}

/**
 * Default pdf-parse page renderer (line-aware), collecting sanitized page texts.
 * @param {string[]} pages
 * @returns {(pageData: object) => Promise<string>}
 */
function makePageCollector(pages) {
  return async function pagerender(pageData) {
    const textContent = await pageData.getTextContent({
      normalizeWhitespace: false,
      disableCombineTextItems: false,
    });
    let lastY;
    let text = '';
    for (const item of textContent.items) {
      if (lastY == item.transform[5] || !lastY) {
        text += item.str;
      } else {
        text += `\n${item.str}`;
      }
      lastY = item.transform[5];
    }
    pages.push(sanitizeExtractedText(text));
    return text;
  };
}

/**
 * @param {string} pdfPath Absolute path to a PDF file.
 * @returns {Promise<string>} Absolute path to the sibling `.md` file.
 */
async function extractPdfToMarkdown(pdfPath) {
  const abs = path.resolve(pdfPath);
  try {
    await fsp.access(abs);
  } catch {
    throw new Error(`PDF file not found: ${pdfPath}`);
  }

  if (!PDF_EXTS.has(extOf(abs))) {
    throw new Error(`Not a PDF file: ${pdfPath}`);
  }

  const mdPath = abs.replace(/\.pdf$/i, '.md');
  try {
    await fsp.access(mdPath);
    return mdPath;
  } catch {
    /* need to extract */
  }

  const buffer = await fsp.readFile(abs);
  const pages = [];
  const data = await pdfParse(buffer, { pagerender: makePageCollector(pages) });

  while (pages.length < (data.numpages || 0)) {
    pages.push('');
  }
  if (pages.length === 0) {
    pages.push('');
  }

  const markdown = pages
    .map((content, i) => `### Page ${i + 1}\n\n${content}`)
    .join('\n\n---\n\n');
  await fsp.writeFile(mdPath, markdown, 'utf8');
  return mdPath;
}

module.exports = {
  extractPdfToMarkdown,
  sanitizeExtractedText,
};
