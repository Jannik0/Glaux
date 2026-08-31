// Shared mutable state that crosses the Models / Sessions / Chat panel
// boundaries. Plain top-level `let` bindings are intentional here (not a
// window.Glaux.* namespace): this app has no bundler, so every <script> in
// index.html shares one global scope, and these few pieces of state are
// read and written from several panel files. Everything that is only used
// within a single panel stays local to that panel's file instead.

/** True once a model has finished loading and is ready to run inference. */
let modelReady = false;

/** The in-flight assistant response stream, if any (see panels/chat.js). */
let activeStream = null;

/** True while a chat request is being sent/streamed. */
let engineBusy = false;

/** True while a model is being loaded into the inference worker (startup, switch, reload). */
let engineLoading = false;

/** Which side panel most recently received a mousedown; used by keyboard shortcuts. */
let activePanel = 'resources';

/** Whether the active engine supports rendering context usage via its chat template. */
let chatTemplateSupported = false;

/** Pipeline tag of the currently loaded engine, if any (see shared/mediaKinds.js ASR handling). */
let activeEnginePipelineTag = null;

/** Human-readable heading for model load / download progress UI. */
let modelProgressHeading = window.Glaux.i18n.t('chat.loadingModel');

/**
 * @param {boolean} loading
 */
function setEngineLoading(loading) {
  if (engineLoading === loading) {
    return;
  }
  engineLoading = loading;
  syncModelPanelDisabled();
  syncSessionsPanelDisabled();
  syncWorkspaceControlsDisabled();
}
