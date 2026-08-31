// Chat panel: message list, composer, attachments, streaming responses, and
// engine status polling.

function t(key, vars) {
  return window.Glaux.i18n.t(key, vars);
}

const messagesEl = document.getElementById('messages');
const loadingMessageEl = document.getElementById('chat-system-status');
const contextUsageEl = document.getElementById('context-usage');
const formEl = document.getElementById('chat-form');
const inputEl = document.getElementById('message-input');
const messageInputBackdropEl = document.getElementById('message-input-backdrop');
const messageInputShellEl = document.getElementById('message-input-shell');
const chatAttachmentsDropEl = document.getElementById('chat-attachments-drop');
const chatAttachmentsPlaceholderEl = document.getElementById('chat-attachments-placeholder');
const chatAttachmentsListEl = document.getElementById('chat-attachments-list');
const sendButtonEl = document.getElementById('send-button');
const reasoningToggleEl = document.getElementById('reasoning-enabled');
const resubmitToggleEl = document.getElementById('resubmit-enabled');
const chatContextMenuEl = document.getElementById('chat-context-menu');

/** @type {Array<{ source: 'resources' | 'outputs', relativePath: string, name: string }>} */
let pendingChatAttachments = [];

/** Stable engine identity for status chat (excludes phase/paging/timing — those change every tick during inference). */
let lastEngineStableSignature = '';

let chatContextTargetIndex = -1;

/** In-memory mirrors of preferences.json toggle values. */
let reasoningEnabledPref = false;
let resubmitEnabledPref = false;

function scrollMessagesToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function scrollMessagesToBottomAfterRender() {
  requestAnimationFrame(scrollMessagesToBottom);
}

function syncReasoningSegmentUI(on) {
  if (!reasoningToggleEl) return;
  reasoningToggleEl.checked = Boolean(on);
}

function getReasoningEnabled() {
  return reasoningEnabledPref;
}

function setReasoningEnabled(on) {
  reasoningEnabledPref = Boolean(on);
  syncReasoningSegmentUI(reasoningEnabledPref);
  if (window.api && typeof window.api.updatePreferences === 'function') {
    void window.api.updatePreferences({ reasoningEnabled: reasoningEnabledPref });
  }
}

function applyReasoningEnabledPreference(on) {
  reasoningEnabledPref = Boolean(on);
  syncReasoningSegmentUI(reasoningEnabledPref);
}

function initializeReasoningToggle() {
  if (!reasoningToggleEl) {
    return;
  }
  syncReasoningSegmentUI(getReasoningEnabled());
  reasoningToggleEl.addEventListener('change', () => {
    setReasoningEnabled(reasoningToggleEl.checked);
  });
}

function getEnableThinkingForRequest() {
  return Boolean(reasoningToggleEl && reasoningToggleEl.checked);
}

function syncResubmitSegmentUI(on) {
  if (!resubmitToggleEl) return;
  resubmitToggleEl.checked = Boolean(on);
}

function getResubmitEnabled() {
  return resubmitEnabledPref;
}

function setResubmitEnabled(on) {
  resubmitEnabledPref = Boolean(on);
  syncResubmitSegmentUI(resubmitEnabledPref);
  if (window.api && typeof window.api.updatePreferences === 'function') {
    void window.api.updatePreferences({ resubmitEnabled: resubmitEnabledPref });
  }
}

function applyResubmitEnabledPreference(on) {
  resubmitEnabledPref = Boolean(on);
  syncResubmitSegmentUI(resubmitEnabledPref);
}

function initializeResubmitToggle() {
  if (!resubmitToggleEl) {
    return;
  }
  syncResubmitSegmentUI(getResubmitEnabled());
  resubmitToggleEl.addEventListener('change', () => {
    setResubmitEnabled(resubmitToggleEl.checked);
    void updateContextUsageIndicator({ refresh: true });
  });
}

function getResubmitForRequest() {
  return Boolean(resubmitToggleEl && resubmitToggleEl.checked);
}

/** @param {number} tokens */
function formatContextTokensCompact(tokens) {
  const n = Number(tokens);
  if (!Number.isFinite(n) || n < 0) {
    return '?';
  }
  if (n >= 1_000_000) {
    const v = n / 1_000_000;
    return v >= 10 ? `${Math.round(v)}M` : `${Number(v.toFixed(1))}M`;
  }
  if (n >= 1000) {
    const v = n / 1000;
    return v >= 10 ? `${Math.round(v)}k` : `${Number(v.toFixed(1))}k`;
  }
  return String(Math.round(n));
}

/** @param {number | null | undefined} used @param {number | null | undefined} total */
function formatContextUsageLabel(used, total) {
  const usedNum = used != null && Number.isFinite(Number(used)) ? Number(used) : 0;
  const hasTotal = total != null && Number.isFinite(Number(total)) && Number(total) > 0;
  const usedLabel = formatContextTokensCompact(usedNum);
  const totalLabel = hasTotal ? formatContextTokensCompact(total) : '?';
  return t('chat.contextUsed', { used: usedLabel, total: totalLabel });
}

function setContextUsageLabel(text) {
  if (!contextUsageEl) {
    return;
  }
  const label = typeof text === 'string' ? text : '';
  contextUsageEl.textContent = label;
  contextUsageEl.hidden = label.length === 0;
}

async function updateContextUsageIndicator(options = {}) {
  const refresh = Boolean(options && options.refresh);
  if (!contextUsageEl || !(window.api && typeof window.api.getContextUsage === 'function')) {
    return;
  }
  if (!modelReady || !chatTemplateSupported) {
    setContextUsageLabel('');
    return;
  }
  try {
    const usage = await window.api.getContextUsage(getResubmitForRequest(), { refresh });
    if (!usage || usage.supported === false) {
      setContextUsageLabel('');
      return;
    }
    const used = usage && usage.used != null ? usage.used : 0;
    const total = usage && usage.total != null ? usage.total : null;
    const stale = Boolean(usage && usage.stale);
    setContextUsageLabel(formatContextUsageLabel(used, total));
    if (contextUsageEl) {
      contextUsageEl.title = stale
        ? t('chat.contextUsageTitleStale')
        : t('chat.contextUsageTitle');
    }
  } catch {
    setContextUsageLabel('');
  }
}

function stopActiveGeneration() {
  if (!activeStream) return;
  activeStream.cancel();
  const thinkingEl = messagesEl.querySelector('.message.assistant.thinking');
  if (thinkingEl) {
    thinkingEl.classList.remove('thinking');
  }
}

/**
 * @param {unknown} status
 */
function applyReasoningSupportFromStatus(status) {
  if (!reasoningToggleEl) {
    return;
  }
  if (engineBusy) {
    reasoningToggleEl.disabled = true;
    return;
  }
  const supported = Boolean(status && status.thinkingSupported);
  if (!supported) {
    reasoningToggleEl.disabled = true;
    reasoningToggleEl.checked = false;
    return;
  }
  reasoningToggleEl.disabled = false;
  syncReasoningSegmentUI(getReasoningEnabled());
}

function applyChatTemplateSupportFromStatus(status) {
  chatTemplateSupported = Boolean(status && status.chatTemplateSupported);
  if (!chatTemplateSupported) {
    setContextUsageLabel('');
  }
}

/**
 * @returns {string | null}
 */
function getActivePipelineTag() {
  if (activeEnginePipelineTag) {
    return activeEnginePipelineTag;
  }
  if (modelsPanelSelectedId && modelPipelineTagsById.has(modelsPanelSelectedId)) {
    return modelPipelineTagsById.get(modelsPanelSelectedId) ?? null;
  }
  return null;
}

async function syncActiveEnginePipelineTag() {
  if (!(window.api && typeof window.api.getStatus === 'function')) {
    return;
  }
  try {
    const status = await window.api.getStatus();
    activeEnginePipelineTag =
      typeof status.pipelineTag === 'string' ? status.pipelineTag : null;
  } catch {
    /* ignore */
  }
}

/**
 * @param {Array<{ source: string, relativePath: string, name?: string }>} attachments
 * @returns {string}
 */
function defaultMessageForAttachments(attachments) {
  const { getMediaKind, ASR_PIPELINE_TAG } = window.Glaux.MediaKinds;
  let hasAudio = false;
  let hasImage = false;
  let hasVideo = false;
  for (const file of attachments || []) {
    const kind = getMediaKind(file.name || file.relativePath);
    if (kind === 'audio') {
      hasAudio = true;
    } else if (kind === 'image') {
      hasImage = true;
    } else if (kind === 'video') {
      hasVideo = true;
    }
  }
  const videoCountsAsAudio =
    getActivePipelineTag() === ASR_PIPELINE_TAG && hasVideo;
  if (hasImage || (hasVideo && !videoCountsAsAudio)) {
    return t('chat.defaultDescribe');
  }
  if (hasAudio || videoCountsAsAudio) {
    return t('chat.defaultTranscribe');
  }
  return '';
}

/**
 * @param {unknown} msg
 * @returns {string}
 */
const CONTEXT_MEDIA_PART_TYPES = new Set(['image', 'audio', 'video']);

function fileNameFromPath(filePath) {
  const normalized = String(filePath).replace(/\\/g, '/');
  const lastSlash = normalized.lastIndexOf('/');
  return lastSlash >= 0 ? normalized.slice(lastSlash + 1) : normalized;
}

/** Document blocks in stored context text: `start <path>` … `end <path>` (see engineManager.js). */
const CONTEXT_DOCUMENT_BLOCK_RE = /^start ([^\n]+)\n([\s\S]*?)^end \1$/gm;

/**
 * @param {string} text
 * @returns {{ text: string, files: Array<{ source: string, relativePath: string, name: string }> }}
 */
function parseContextDocumentBlocks(text) {
  if (!text || typeof text !== 'string') {
    return { text: '', files: [] };
  }
  const files = [];
  const blocks = [];
  const re = new RegExp(CONTEXT_DOCUMENT_BLOCK_RE.source, 'gm');
  let match;
  while ((match = re.exec(text)) !== null) {
    const relativePath = match[1].trim();
    if (!relativePath) continue;
    blocks.push(match[0]);
    files.push({
      source: 'resources',
      relativePath,
      name: fileNameFromPath(relativePath),
    });
  }
  let displayText = text;
  for (const block of blocks) {
    displayText = displayText.replace(block, '');
  }
  return { text: displayText.trim(), files };
}

function contextMessageToDisplayText(msg) {
  if (!msg || typeof msg !== 'object') {
    return '';
  }
  const content = /** @type {{ content?: unknown }} */ (msg).content;
  if (typeof content === 'string') {
    return parseContextDocumentBlocks(content).text;
  }
  if (!Array.isArray(content)) {
    return '';
  }
  const raw = content
    .filter((part) => part && typeof part === 'object' && part.type === 'text')
    .map((part) => String(part.text || ''))
    .join('\n');
  return parseContextDocumentBlocks(raw).text;
}

/**
 * @param {unknown} msg
 * @returns {Array<{ source: string, relativePath: string, name: string }>}
 */
function contextMessageToDisplayFiles(msg) {
  if (!msg || typeof msg !== 'object' || /** @type {{ role?: string }} */ (msg).role !== 'user') {
    return [];
  }
  const content = /** @type {{ content?: unknown }} */ (msg).content;
  const files = [];
  if (Array.isArray(content)) {
    const raw = content
      .filter((part) => part && typeof part === 'object' && part.type === 'text')
      .map((part) => String(part.text || ''))
      .join('\n');
    files.push(...parseContextDocumentBlocks(raw).files);
    for (const part of content) {
      if (!part || typeof part !== 'object') {
        continue;
      }
      const typed = /** @type {{ type?: string, path?: unknown, source?: unknown, relativePath?: unknown }} */ (
        part
      );
      if (!CONTEXT_MEDIA_PART_TYPES.has(typed.type)) {
        continue;
      }
      if (typeof typed.source === 'string' && typeof typed.relativePath === 'string' && typed.relativePath) {
        const source = typed.source === 'outputs' ? 'outputs' : 'resources';
        files.push({
          source,
          relativePath: typed.relativePath,
          name: fileNameFromPath(typed.relativePath),
        });
        continue;
      }
      if (typeof typed.path !== 'string' || !typed.path.trim()) {
        continue;
      }
      const name = fileNameFromPath(typed.path);
      files.push({
        source: 'resources',
        relativePath: name,
        name,
      });
    }
  }
  return files;
}

function clearChatMessages() {
  messagesEl.querySelectorAll('.message:not(.system)').forEach((el) => {
    el.remove();
  });
}

/**
 * @param {Array<{ role?: string, content?: unknown }>} messages
 */
function rebuildChatFromContext(messages) {
  for (const msg of messages) {
    const role = msg && msg.role;
    if (role !== 'user' && role !== 'assistant') {
      continue;
    }
    const text = contextMessageToDisplayText(msg);
    const files = role === 'user' ? contextMessageToDisplayFiles(msg) : [];
    appendMessage(role, text, files);
  }
}

function getChatMessageElements() {
  return [...messagesEl.querySelectorAll('.message.user, .message.assistant')];
}

function getChatMessageContextIndex(messageEl) {
  if (!messageEl) {
    return -1;
  }
  let contextIndex = -1;
  for (const el of getChatMessageElements()) {
    contextIndex += 1;
    if (el === messageEl) {
      return contextIndex;
    }
  }
  return -1;
}

function markStoppedTurn(assistantContainer, userMessageEl) {
  assistantContainer.classList.add('stopped');
  if (userMessageEl) {
    userMessageEl.classList.add('stopped');
  }
}

function isGenerationCanceledError(err) {
  if (!err) {
    return false;
  }
  if (err.code === 'E_CANCELED' || err.name === 'E_CANCELED') {
    return true;
  }
  const message = String(err.message || '');
  if (/generation canceled/i.test(message)) {
    return true;
  }
  return message === t('preload.generationCanceled');
}

function closeChatContextMenu() {
  if (!chatContextMenuEl) {
    return;
  }
  chatContextMenuEl.classList.add('hidden');
  chatContextMenuEl.textContent = '';
  chatContextTargetIndex = -1;
}

function addChatContextMenuItem(label, onClick) {
  if (!chatContextMenuEl) {
    return;
  }
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'context-menu-item';
  button.textContent = label;
  button.addEventListener('click', async (event) => {
    event.preventDefault();
    closeChatContextMenu();
    await onClick();
  });
  chatContextMenuEl.appendChild(button);
}

function openChatContextMenu(x, y) {
  if (!chatContextMenuEl || chatContextTargetIndex < 0 || activeStream) {
    return;
  }

  chatContextMenuEl.textContent = '';
  const messageIndex = chatContextTargetIndex;

  addChatContextMenuItem(t('chat.delete'), async () => {
    await deleteChatMessageAtIndex(messageIndex);
  });
  addChatContextMenuItem(t('chat.export'), async () => {
    await exportChatMessageAtIndex(messageIndex);
  });

  if (!chatContextMenuEl.children.length) {
    closeChatContextMenu();
    return;
  }

  chatContextMenuEl.classList.remove('hidden');
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const menuWidth = 180;
  const menuHeight = chatContextMenuEl.offsetHeight || 80;
  const left = Math.min(x, viewportWidth - menuWidth - 8);
  const top = Math.min(y, viewportHeight - menuHeight - 8);
  chatContextMenuEl.style.left = `${Math.max(8, left)}px`;
  chatContextMenuEl.style.top = `${Math.max(8, top)}px`;
}

async function deleteChatMessageAtIndex(messageIndex) {
  if (!window.api || !window.api.sessions || typeof window.api.sessions.deleteMessage !== 'function') {
    return;
  }
  try {
    const messages = await window.api.sessions.deleteMessage(messageIndex);
    clearChatMessages();
    rebuildChatFromContext(messages);
    void updateContextUsageIndicator({ refresh: true });
  } catch (err) {
    setLoadingStatusMessage(t('chat.couldNotDeleteMessage', { message: err.message || String(err) }));
  }
}

async function exportChatMessageAtIndex(messageIndex) {
  if (!window.api || !window.api.chat || typeof window.api.chat.exportMessage !== 'function') {
    return;
  }
  try {
    const result = await window.api.chat.exportMessage(messageIndex);
    if (result.tree) {
      const outputsPanel = window.Glaux.Outputs.panel;
      outputsPanel.tree = result.tree;
      outputsPanel.rebuildNodeIndex();
      outputsPanel.render();
    }
    window.Glaux.Outputs.panel.setStatus(t('panels.outputs.exported', { fileName: result.fileName }), false);
  } catch (err) {
    window.Glaux.Outputs.panel.setStatus(err.message || String(err), true);
  }
}

function initializeChatMessageContextMenu() {
  if (!messagesEl || !chatContextMenuEl) {
    return;
  }

  messagesEl.addEventListener('contextmenu', (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }
    const messageEl = event.target.closest('.message.user, .message.assistant');
    if (
      !messageEl ||
      messageEl.classList.contains('thinking') ||
      activeStream
    ) {
      return;
    }

    const messageIndex = getChatMessageContextIndex(messageEl);
    if (messageIndex < 0) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    window.Glaux.Resources.panel.closeContextMenu();
    window.Glaux.Outputs.panel.closeContextMenu();
    chatContextTargetIndex = messageIndex;
    openChatContextMenu(event.clientX, event.clientY);
  });
}

function createAssistantMessageStructure() {
  const container = document.createElement('div');
  container.classList.add('message', 'assistant');

  const details = document.createElement('details');
  details.classList.add('thoughts-section');
  details.style.display = 'none';

  const summary = document.createElement('summary');
  summary.textContent = t('chat.thoughts');
  details.appendChild(summary);

  const thoughtsContent = document.createElement('div');
  thoughtsContent.classList.add('thoughts-content');
  details.appendChild(thoughtsContent);

  const answerContent = document.createElement('div');
  answerContent.classList.add('answer-content');

  container.appendChild(details);
  container.appendChild(answerContent);

  return { container, details, thoughtsContent, answerContent };
}

/**
 * @param {string} role
 * @param {string} text
 * @param {Array<{ source: string, relativePath: string, name: string }>} [files]
 */
function appendMessage(role, text, files = []) {
  const { renderFormattedMessage, parseTagsAndAnswer } = window.Glaux.ThinkingParse;
  if (role === 'assistant' && text) {
    const { thinking, answer } = parseTagsAndAnswer(text);
    if (thinking) {
      const { container, details, thoughtsContent, answerContent } = createAssistantMessageStructure();
      details.style.display = '';
      renderFormattedMessage(thoughtsContent, thinking);
      renderFormattedMessage(answerContent, answer);
      messagesEl.appendChild(container);
      scrollMessagesToBottom();
      return container;
    }
  }
  const div = document.createElement('div');
  div.classList.add('message', role);
  const bodyEl = document.createElement('div');
  bodyEl.className = 'message-body';
  renderFormattedMessage(bodyEl, text);
  div.appendChild(bodyEl);
  if (role === 'user' && files.length > 0) {
    const attachmentsEl = document.createElement('div');
    attachmentsEl.className = 'message-attachments';
    for (const file of files) {
      const chip = document.createElement('span');
      chip.className = 'message-attachment-chip';
      const iconEl = document.createElement('span');
      iconEl.className = 'tree-icon';
      iconEl.textContent = window.Glaux.MediaKinds.getFileIcon(file.name);
      const nameEl = document.createElement('span');
      nameEl.className = 'message-attachment-name';
      nameEl.textContent = file.name;
      chip.append(iconEl, nameEl);
      attachmentsEl.appendChild(chip);
    }
    div.appendChild(attachmentsEl);
  }
  messagesEl.appendChild(div);
  scrollMessagesToBottom();
  return div;
}

function updateSendButton() {
  const iconEl = sendButtonEl.querySelector('.icon');
  const labelEl = sendButtonEl.querySelector('.icon + span');
  if (engineBusy && activeStream) {
    sendButtonEl.disabled = false;
    if (iconEl) iconEl.textContent = '\u25A0';
    if (labelEl) labelEl.textContent = t('chat.stop');
    return;
  }
  sendButtonEl.disabled = !modelReady || engineBusy || engineLoading;
  if (iconEl) iconEl.textContent = '\u27A4';
  if (labelEl) labelEl.textContent = t('chat.send');
}

function setReady(isReady) {
  // pollStatus reports ready=false while generating; keep modelReady until the stream ends.
  if (isReady || !engineBusy) {
    modelReady = isReady;
  }
  if (isReady) {
    setEngineLoading(false);
  }
  inputEl.disabled = !modelReady || engineBusy || engineLoading;
  updateSendButton();
  syncModelPanelDisabled();
}

function setBusy(isBusy) {
  engineBusy = isBusy;
  inputEl.disabled = isBusy || !modelReady || engineLoading;
  if (reasoningToggleEl) reasoningToggleEl.disabled = isBusy;
  if (resubmitToggleEl) resubmitToggleEl.disabled = isBusy;
  syncModelPanelDisabled();
  syncSessionsPanelDisabled();
  syncWorkspaceControlsDisabled();
  updateSendButton();
}

function isInferenceAttachableNode(node) {
  if (!node || node.type !== 'file') return false;
  const { IMAGE_EXTS, AUDIO_EXTS, MEDIA_VIDEO_EXTS, DOCUMENT_EXTS } = window.Glaux.MediaKinds;
  const dotIdx = node.name.lastIndexOf('.');
  const ext = dotIdx >= 0 ? node.name.slice(dotIdx + 1).toLowerCase() : '';
  return (
    IMAGE_EXTS.has(ext) ||
    AUDIO_EXTS.has(ext) ||
    MEDIA_VIDEO_EXTS.has(ext) ||
    DOCUMENT_EXTS.has(ext)
  );
}

function updateMessageInputHighlight() {
  if (!messageInputBackdropEl || !inputEl) return;
  messageInputBackdropEl.textContent = inputEl.value;
  syncMessageInputScroll();
}

function syncMessageInputScroll() {
  if (!messageInputBackdropEl || !inputEl) return;
  messageInputBackdropEl.scrollTop = inputEl.scrollTop;
  messageInputBackdropEl.scrollLeft = inputEl.scrollLeft;
}

/**
 * @param {string} resourcePath
 * @param {string} outputPath
 * @returns {{ source: 'resources' | 'outputs', relativePath: string, name: string } | null}
 */
function resolveDroppedTreeFile(resourcePath, outputPath) {
  if (resourcePath) {
    const node = window.Glaux.Resources.panel.nodesByPath.get(resourcePath);
    if (!isInferenceAttachableNode(node)) return null;
    return {
      source: 'resources',
      relativePath: node.relativePath,
      name: node.name,
    };
  }
  if (outputPath) {
    const node = window.Glaux.Outputs.panel.nodesByPath.get(outputPath);
    if (!isInferenceAttachableNode(node)) return null;
    return {
      source: 'outputs',
      relativePath: node.relativePath,
      name: node.name,
    };
  }
  return null;
}

function chatAttachmentKey(file) {
  return `${file.source}:${file.relativePath}`;
}

function renderChatAttachments() {
  if (!chatAttachmentsDropEl || !chatAttachmentsListEl || !chatAttachmentsPlaceholderEl) {
    return;
  }
  chatAttachmentsListEl.innerHTML = '';
  const isEmpty = pendingChatAttachments.length === 0;
  chatAttachmentsDropEl.classList.toggle('empty', isEmpty);
  chatAttachmentsPlaceholderEl.classList.toggle('hidden', !isEmpty);
  chatAttachmentsListEl.classList.toggle('hidden', isEmpty);

  for (const file of pendingChatAttachments) {
    const chip = document.createElement('span');
    chip.className = 'chat-attachment-chip';
    chip.title = file.relativePath;

    const iconEl = document.createElement('span');
    iconEl.className = 'tree-icon';
    iconEl.textContent = window.Glaux.MediaKinds.getFileIcon(file.name);

    const nameEl = document.createElement('span');
    nameEl.className = 'chat-attachment-name';
    nameEl.textContent = file.name;

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'chat-attachment-remove';
    removeBtn.setAttribute('aria-label', t('chat.removeAttachmentAria', { name: file.name }));
    removeBtn.textContent = '\u00D7';
    removeBtn.addEventListener('click', () => {
      removeChatAttachment(file);
    });

    chip.append(iconEl, nameEl, removeBtn);
    chatAttachmentsListEl.appendChild(chip);
  }
}

function addChatAttachment(file) {
  if (!file || !file.relativePath) return;
  const key = chatAttachmentKey(file);
  if (pendingChatAttachments.some((entry) => chatAttachmentKey(entry) === key)) {
    return;
  }
  pendingChatAttachments.push(file);
  renderChatAttachments();
}

function removeChatAttachment(file) {
  const key = chatAttachmentKey(file);
  pendingChatAttachments = pendingChatAttachments.filter(
    (entry) => chatAttachmentKey(entry) !== key
  );
  renderChatAttachments();
}

function clearChatAttachments() {
  pendingChatAttachments = [];
  renderChatAttachments();
}

function getPendingChatAttachmentsPayload() {
  return pendingChatAttachments.map(({ source, relativePath }) => ({
    source,
    relativePath,
  }));
}

function getPendingChatAttachmentsForDisplay() {
  return pendingChatAttachments.map(({ source, relativePath, name }) => ({
    source,
    relativePath,
    name,
  }));
}

/** @returns {'none' | 'accept' | 'reject'} */
function getChatAttachmentDragState(dataTransfer) {
  const { resourcePath, outputPath } = readDroppedTreePaths(dataTransfer);
  if (!resourcePath && !outputPath) {
    return 'none';
  }
  return resolveDroppedTreeFile(resourcePath, outputPath) ? 'accept' : 'reject';
}

function shouldRejectInputFileDrop(dataTransfer) {
  if (!dataTransfer) return false;
  if (window.Glaux.Resources.panel.draggedPath || window.Glaux.Outputs.panel.draggedPath) return true;
  if (!dataTransfer.types) return false;
  const types = Array.from(dataTransfer.types);
  if (types.includes('Files')) return true;
  if (types.includes('text/x-resource-path') || types.includes('text/x-output-path')) {
    return true;
  }
  if (types.includes('text/uri-list')) return true;
  return false;
}

function initializeDropReject(element) {
  if (!element) return;

  const rejectDrop = (event) => {
    if (!shouldRejectInputFileDrop(event.dataTransfer)) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.type === 'dragover') {
      event.dataTransfer.dropEffect = 'none';
    }
  };

  element.addEventListener('dragover', rejectDrop, true);
  element.addEventListener('drop', rejectDrop, true);
}

function readDroppedTreePaths(dataTransfer) {
  let resourcePath = dataTransfer ? dataTransfer.getData('text/x-resource-path') : '';
  let outputPath = dataTransfer ? dataTransfer.getData('text/x-output-path') : '';
  if (!resourcePath && window.Glaux.Resources.panel.draggedPath) {
    resourcePath = window.Glaux.Resources.panel.draggedPath;
  }
  if (!outputPath && window.Glaux.Outputs.panel.draggedPath) {
    outputPath = window.Glaux.Outputs.panel.draggedPath;
  }
  return { resourcePath, outputPath };
}

function initializeChatAttachmentsDrop() {
  const dropEl = chatAttachmentsDropEl;
  if (!dropEl) return;

  const setDragHighlight = (state) => {
    dropEl.classList.toggle('drag-over', state === 'accept');
    dropEl.classList.toggle('drag-over-reject', state === 'reject');
  };

  const onDragOver = (event) => {
    const state = getChatAttachmentDragState(event.dataTransfer);
    if (state === 'none') return;
    event.preventDefault();
    event.dataTransfer.dropEffect = state === 'accept' ? 'copy' : 'none';
    setDragHighlight(state);
  };

  const onDragEnter = (event) => {
    const state = getChatAttachmentDragState(event.dataTransfer);
    if (state === 'none') return;
    event.preventDefault();
    setDragHighlight(state);
  };

  const onDragLeave = (event) => {
    const related = event.relatedTarget;
    if (related instanceof Node && dropEl.contains(related)) return;
    setDragHighlight('none');
  };

  const onDrop = (event) => {
    const state = getChatAttachmentDragState(event.dataTransfer);
    setDragHighlight('none');
    if (state !== 'accept') {
      if (state === 'reject') {
        event.preventDefault();
        event.stopPropagation();
      }
      return;
    }
    const { resourcePath, outputPath } = readDroppedTreePaths(event.dataTransfer);
    const file = resolveDroppedTreeFile(resourcePath, outputPath);
    if (!file) return;

    event.preventDefault();
    event.stopPropagation();
    addChatAttachment(file);
  };

  dropEl.addEventListener('dragover', onDragOver);
  dropEl.addEventListener('dragenter', onDragEnter);
  dropEl.addEventListener('dragleave', onDragLeave);
  dropEl.addEventListener('drop', onDrop);
  dropEl.addEventListener('dragend', () => {
    setDragHighlight('none');
  });
}

formEl.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (activeStream) {
    stopActiveGeneration();
    return;
  }
  const raw = inputEl.value;
  const attachmentPayload = getPendingChatAttachmentsPayload();
  const attachmentDisplay = getPendingChatAttachmentsForDisplay();
  if ((!raw.trim() && attachmentPayload.length === 0) || !modelReady) return;
  let text = raw.replace(/\s+$/, '');
  if (!text.trim() && attachmentPayload.length > 0) {
    await syncActiveEnginePipelineTag();
    const defaultText = defaultMessageForAttachments(attachmentPayload);
    if (defaultText) {
      text = defaultText;
    }
  }

  const userMessageEl = appendMessage('user', text, attachmentDisplay);
  inputEl.value = '';
  clearChatAttachments();
  updateMessageInputHighlight();

  const enableThinking = getEnableThinkingForRequest();
  const resubmit = getResubmitForRequest();
  const files = attachmentPayload;

  const { container, details, thoughtsContent, answerContent } = createAssistantMessageStructure();
  container.classList.add('thinking');
  messagesEl.appendChild(container);
  scrollMessagesToBottomAfterRender();

  const { renderFormattedMessage, parseTagsAndAnswer, createTagStreamParser } = window.Glaux.ThinkingParse;

  let streamed = '';
  let streamParser = null;
  let streamStartsInThinking = false;
  const ensureStreamParser = () => {
    if (!streamParser) {
      streamParser = createTagStreamParser(thoughtsContent, answerContent, details, {
        startInThinking: streamStartsInThinking,
      });
    }
  };
  try {
    const stream = window.api.sendMessageStream(text, {
      enableThinking,
      resubmit,
      files,
      onStarted: (meta) => {
        streamStartsInThinking = Boolean(meta && meta.startsInThinking);
        ensureStreamParser();
      },
      onToken: (chunk) => {
        ensureStreamParser();
        streamed += chunk;
        streamParser(chunk);
      },
      onSnapshot: (text) => {
        // ASR (and similar) send full hypothesis snapshots; replace the bubble
        // instead of appending so rewrites do not loop and progress continues.
        streamParser = null;
        streamed = typeof text === 'string' ? text : '';
        details.style.display = 'none';
        thoughtsContent.textContent = '';
        answerContent.textContent = '';
        renderFormattedMessage(answerContent, streamed);
        scrollMessagesToBottomAfterRender();
      },
    });
    activeStream = stream;
    setBusy(true);
    const response = await stream.completion;
    if (streamParser) {
      streamParser.flush();
    }
    // If streamed chunks drifted from the authoritative final (e.g. ASR
    // hypothesis rewrites), replace the live answer with the final text.
    if (typeof response === 'string' && response !== streamed) {
      const { thinking, answer } = parseTagsAndAnswer(response);
      if (thinking) {
        details.style.display = '';
        thoughtsContent.textContent = '';
        renderFormattedMessage(thoughtsContent, thinking);
      }
      answerContent.textContent = '';
      renderFormattedMessage(answerContent, answer || response);
      streamed = response;
    }
    container.classList.remove('thinking');
  } catch (err) {
    const canceled = isGenerationCanceledError(err);
    if (canceled) {
      const stopMarker = '[STOP]';
      const afterFlush =
        streamParser && typeof streamParser.flush === 'function'
          ? streamParser.flush()
          : { state: 'detect' };
      const endState = afterFlush && afterFlush.state;

      if (endState === 'thinking') {
        // flush() already committed partial thinking to the thoughts panel; only show [STOP] in the answer.
        details.style.display = '';
        answerContent.textContent = '';
        renderFormattedMessage(answerContent, stopMarker);
      } else if (endState === 'answering') {
        const body = (afterFlush && afterFlush.answer) || answerContent.textContent || '';
        answerContent.textContent = '';
        renderFormattedMessage(answerContent, body ? `${body}\n\n${stopMarker}` : stopMarker);
      } else {
        const fullText = streamed ? `${streamed}\n\n${stopMarker}` : stopMarker;
        const { thinking, answer } = parseTagsAndAnswer(fullText);
        if (thinking) {
          details.style.display = '';
          thoughtsContent.textContent = '';
          renderFormattedMessage(thoughtsContent, thinking);
        }
        answerContent.textContent = '';
        renderFormattedMessage(answerContent, answer);
      }
      container.classList.remove('thinking');
      markStoppedTurn(container, userMessageEl);
    } else {
      answerContent.textContent = '';
      renderFormattedMessage(answerContent, t('chat.errorPrefix', { message: err.message || String(err) }));
      container.classList.remove('thinking');
    }
  } finally {
    activeStream = null;
    setBusy(false);
    scrollMessagesToBottom();
    void refreshSessionsList();
    void updateContextUsageIndicator();
  }
});

sendButtonEl.addEventListener('click', () => {
  if (activeStream) {
    stopActiveGeneration();
    return;
  }
  if (typeof formEl.requestSubmit === 'function') {
    formEl.requestSubmit();
  } else {
    formEl.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
  }
});

inputEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    if (activeStream) {
      return;
    }
    if (typeof formEl.requestSubmit === 'function') {
      formEl.requestSubmit();
    } else {
      formEl.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    }
  }
});

inputEl.addEventListener('input', () => {
  updateMessageInputHighlight();
});

inputEl.addEventListener('scroll', () => {
  syncMessageInputScroll();
});

/**
 * @param {string} text
 * @returns {string}
 */
function formatEngineStatusSentence(text) {
  const trimmed = String(text || '').trim();
  if (!trimmed) {
    return '';
  }
  if (/[.!?%]$/.test(trimmed)) {
    return trimmed;
  }
  return `${trimmed}.`;
}

function formatModelReadyStatusMessage() {
  return t('chat.modelReady');
}

const ENGINE_STATUS_CLEAR_MS = 5000;
/** @type {ReturnType<typeof setTimeout> | null} */
let engineStatusClearTimer = null;

function clearEngineStatusClearTimer() {
  if (!engineStatusClearTimer) {
    return;
  }
  clearTimeout(engineStatusClearTimer);
  engineStatusClearTimer = null;
}

function isModelReadyStatusMessage(formatted) {
  return formatted === formatEngineStatusSentence(formatModelReadyStatusMessage());
}

/**
 * Engine status floating over the top of the transcript.
 * "Model ready" auto-clears after a short delay; other statuses stay until replaced.
 * @param {string} text
 */
function setLoadingStatusMessage(text) {
  if (!loadingMessageEl) {
    return;
  }
  clearEngineStatusClearTimer();
  const formatted = formatEngineStatusSentence(text);
  window.Glaux.ThinkingParse.renderFormattedMessage(loadingMessageEl, formatted);
  loadingMessageEl.hidden = !formatted;
  if (!isModelReadyStatusMessage(formatted)) {
    return;
  }
  engineStatusClearTimer = setTimeout(() => {
    engineStatusClearTimer = null;
    setLoadingStatusMessage('');
  }, ENGINE_STATUS_CLEAR_MS);
}

async function pollStatus() {
  if (!(window.api && typeof window.api.getStatus === 'function')) {
    return;
  }

  try {
    const status = await window.api.getStatus();
    activeEnginePipelineTag =
      typeof status.pipelineTag === 'string' ? status.pipelineTag : null;
    applyReasoningSupportFromStatus(status);
    applyChatTemplateSupportFromStatus(status);
    setEngineLoading(status.phase === 'loading');
    const isReady = Boolean(status.ready);
    setReady(isReady);
    const stableSignature = JSON.stringify({
      ready: isReady,
      modelPath: status.modelPath,
      thinkingSupported: Boolean(status.thinkingSupported),
      chatTemplateSupported: Boolean(status.chatTemplateSupported),
    });
    if (stableSignature !== lastEngineStableSignature) {
      lastEngineStableSignature = stableSignature;
      void updateContextUsageIndicator({ refresh: isReady });
    }
    if (status.ready) {
      setReady(true);
      setTimeout(pollStatus, 2000);
      return;
    }
  } catch (_err) {}

  setTimeout(pollStatus, 2000);
}

if (loadingMessageEl) {
  loadingMessageEl.hidden = true;
}
setReady(false);

function reportEngineLoadError(message) {
  const msg = message || t('chat.unknownError');
  hideModelLoadModal();
  setLoadingStatusMessage(t('chat.failedToLoadModel', { message: msg }));
}

function showNoModelLoadedMessage() {
  modelProgressHeading = t('chat.noModelLoadedHeading');
  setLoadingStatusMessage(t('chat.noModelLoadedBody'));
}
