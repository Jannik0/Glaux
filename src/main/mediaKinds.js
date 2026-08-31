const MEDIA_IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp']);
const MEDIA_AUDIO_EXTS = new Set(['wav', 'mp3', 'ogg', 'flac', 'm4a']);
const MEDIA_VIDEO_EXTS = new Set(['mp4', 'mov', 'avi', 'mkv', 'webm', 'mts', 'm2ts', 'ts']);
const MEDIA_PDF_EXTS = new Set(['pdf']);

function getMediaKindFromFileName(fileName) {
  const lowerName = String(fileName || '').toLowerCase();
  const dotIndex = lowerName.lastIndexOf('.');
  const ext = dotIndex === -1 ? '' : lowerName.slice(dotIndex + 1);
  if (MEDIA_IMAGE_EXTS.has(ext)) {
    return 'image';
  }
  if (MEDIA_AUDIO_EXTS.has(ext)) {
    return 'audio';
  }
  if (MEDIA_VIDEO_EXTS.has(ext)) {
    return 'video';
  }
  if (MEDIA_PDF_EXTS.has(ext)) {
    return 'pdf';
  }
  return null;
}

module.exports = {
  MEDIA_IMAGE_EXTS,
  MEDIA_AUDIO_EXTS,
  MEDIA_VIDEO_EXTS,
  MEDIA_PDF_EXTS,
  getMediaKindFromFileName,
};
