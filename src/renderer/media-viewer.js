function t(key, vars) {
  return window.Glaux.i18n.t(key, vars);
}

const titleEl = document.getElementById('viewer-title');
const contentEl = document.getElementById('viewer-content');
const loadingEl = document.getElementById('viewer-loading');
const errorEl = document.getElementById('viewer-error');
const imageEl = document.getElementById('media-image');
const audioEl = document.getElementById('media-audio');
const videoEl = document.getElementById('media-video');
const pdfEl = document.getElementById('media-pdf');

function showError(message) {
  loadingEl.classList.add('hidden');
  imageEl.classList.add('hidden');
  audioEl.classList.add('hidden');
  videoEl.classList.add('hidden');
  pdfEl.classList.add('hidden');
  errorEl.textContent = message;
  errorEl.classList.remove('hidden');
}

function initializeMedia(context) {
  const { fileName, kind, mediaUrl } = context;
  titleEl.textContent = fileName;
  document.title = `${fileName} - Glaux`;
  loadingEl.classList.add('hidden');
  errorEl.classList.add('hidden');
  contentEl.classList.remove('image-mode', 'audio-mode', 'video-mode', 'pdf-mode');

  if (kind === 'image') {
    contentEl.classList.add('image-mode');
    imageEl.alt = fileName;
    imageEl.src = mediaUrl;
    imageEl.classList.remove('hidden');
    imageEl.addEventListener(
      'error',
      () => {
        showError(t('viewer.couldNotLoadImage'));
      },
      { once: true }
    );
    return;
  }

  if (kind === 'audio') {
    contentEl.classList.add('audio-mode');
    audioEl.src = mediaUrl;
    audioEl.classList.remove('hidden');
    audioEl.addEventListener(
      'error',
      () => {
        showError(t('viewer.couldNotLoadAudio'));
      },
      { once: true }
    );
    return;
  }

  if (kind === 'video') {
    contentEl.classList.add('video-mode');
    videoEl.src = mediaUrl;
    videoEl.classList.remove('hidden');
    videoEl.addEventListener(
      'error',
      () => {
        showError(t('viewer.couldNotLoadVideo'));
      },
      { once: true }
    );
    return;
  }

  if (kind === 'pdf') {
    contentEl.classList.add('pdf-mode');
    pdfEl.title = fileName;
    pdfEl.src = mediaUrl;
    pdfEl.classList.remove('hidden');
    return;
  }

  showError(t('viewer.unsupportedMediaType'));
}

async function initializeViewer() {
  try {
    const context = await window.mediaApi.getContext();
    initializeMedia(context);
  } catch (err) {
    showError(err.message || String(err));
  }
}

void initializeViewer();
