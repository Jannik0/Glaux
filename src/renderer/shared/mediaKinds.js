// File-kind detection and "open in editor/viewer" helpers shared by the
// Resources and Outputs tree panels. Exposed as window.Glaux.MediaKinds.
(function () {
  window.Glaux = window.Glaux || {};

  const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp']);
  const AUDIO_EXTS = new Set(['wav', 'mp3', 'ogg', 'flac', 'm4a']);
  const DOCUMENT_EXTS = new Set(['md', 'txt', 'pdf']);
  const MEDIA_IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp']);
  const MEDIA_VIDEO_EXTS = new Set(['mp4', 'mov', 'avi', 'mkv', 'webm', 'mts', 'm2ts', 'ts']);
  const MEDIA_PDF_EXTS = new Set(['pdf']);
  const ASR_PIPELINE_TAG = 'automatic-speech-recognition';

  function getFileNameParts(fileName) {
    const name = String(fileName || '');
    const lastDotIndex = name.lastIndexOf('.');
    if (lastDotIndex <= 0 || lastDotIndex === name.length - 1) {
      return { baseName: name, extension: '' };
    }
    return {
      baseName: name.slice(0, lastDotIndex),
      extension: name.slice(lastDotIndex),
    };
  }

  function isMarkdownFile(name) {
    const lowerName = String(name || '').toLowerCase();
    return lowerName.endsWith('.md') || lowerName.endsWith('.txt');
  }

  function getMediaKind(name) {
    const lowerName = String(name || '').toLowerCase();
    const dotIndex = lowerName.lastIndexOf('.');
    const ext = dotIndex === -1 ? '' : lowerName.slice(dotIndex + 1);
    if (MEDIA_IMAGE_EXTS.has(ext)) {
      return 'image';
    }
    if (AUDIO_EXTS.has(ext)) {
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

  function isOpenableMediaFile(name) {
    return getMediaKind(name) !== null;
  }

  function getFileIcon(name) {
    const lowerName = (name || '').toLowerCase();
    const dotIndex = lowerName.lastIndexOf('.');
    const ext = dotIndex === -1 ? '' : lowerName.slice(dotIndex + 1);

    switch (ext) {
      case 'png':
      case 'jpg':
      case 'jpeg':
      case 'gif':
      case 'webp':
      case 'svg':
        return '🖼️';
      case 'mp4':
      case 'mov':
      case 'avi':
      case 'mkv':
      case 'webm':
      case 'mts':
      case 'm2ts':
      case 'ts':
        return '🎞️';
      case 'mp3':
      case 'wav':
      case 'flac':
      case 'ogg':
        return '🎵';
      case 'pdf':
        return '📕';
      case 'md':
      case 'txt':
      case 'rtf':
        return '📄';
      case 'json':
      case 'yaml':
      case 'yml':
      case 'xml':
      case 'toml':
        return '🧩';
      case 'js':
      case 'tsx':
      case 'jsx':
      case 'py':
      case 'cs':
      case 'java':
      case 'cpp':
      case 'c':
      case 'go':
      case 'rs':
        return '💻';
      case 'zip':
      case 'rar':
      case '7z':
      case 'tar':
      case 'gz':
        return '🗜️';
      default:
        return '📄';
    }
  }

  /** @type {Map<string, (message: string, isError?: boolean) => void>} */
  const statusSettersByPanel = new Map();

  /**
   * Registered lazily by panels/resources.js and panels/outputs.js so this
   * shared module never has to import panel-specific code.
   * @param {'resources' | 'outputs'} panel
   * @param {(message: string, isError?: boolean) => void} setStatusFn
   */
  function registerStatusSetter(panel, setStatusFn) {
    statusSettersByPanel.set(panel, setStatusFn);
  }

  function reportPanelError(panel, err) {
    const setStatus = statusSettersByPanel.get(panel);
    if (setStatus) {
      setStatus(err.message || String(err), true);
    }
  }

  async function openMarkdownEditor(panel, relativePath) {
    if (!relativePath || !isMarkdownFile(relativePath.split('/').pop())) {
      return;
    }
    try {
      await window.api.markdown.openEditor(panel, relativePath);
    } catch (err) {
      reportPanelError(panel, err);
    }
  }

  async function openMediaViewer(panel, relativePath) {
    const fileName = relativePath ? relativePath.split('/').pop() : '';
    if (!relativePath || !isOpenableMediaFile(fileName)) {
      return;
    }
    try {
      await window.api.media.openViewer(panel, relativePath);
    } catch (err) {
      reportPanelError(panel, err);
    }
  }

  function tryOpenTreeFile(node, panel) {
    if (!node || node.type !== 'file') {
      return false;
    }
    if (isMarkdownFile(node.name)) {
      void openMarkdownEditor(panel, node.relativePath);
      return true;
    }
    if (isOpenableMediaFile(node.name)) {
      void openMediaViewer(panel, node.relativePath);
      return true;
    }
    return false;
  }

  window.Glaux.MediaKinds = {
    IMAGE_EXTS,
    AUDIO_EXTS,
    DOCUMENT_EXTS,
    MEDIA_IMAGE_EXTS,
    MEDIA_VIDEO_EXTS,
    MEDIA_PDF_EXTS,
    ASR_PIPELINE_TAG,
    getFileNameParts,
    isMarkdownFile,
    getMediaKind,
    isOpenableMediaFile,
    getFileIcon,
    registerStatusSetter,
    openMarkdownEditor,
    openMediaViewer,
    tryOpenTreeFile,
  };
})();
