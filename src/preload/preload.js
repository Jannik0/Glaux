const { contextBridge, ipcRenderer, webUtils } = require('electron');

let cachedI18n = null;

function lookupCatalogString(catalog, key) {
  if (!catalog || typeof catalog !== 'object' || typeof key !== 'string' || !key) {
    return undefined;
  }
  let node = catalog;
  for (const part of key.split('.')) {
    if (!node || typeof node !== 'object' || !(part in node)) {
      return undefined;
    }
    node = node[part];
  }
  return typeof node === 'string' ? node : undefined;
}

function getI18nBundle() {
  if (cachedI18n) {
    return cachedI18n;
  }
  try {
    cachedI18n = ipcRenderer.sendSync('i18n:get');
  } catch {
    cachedI18n = null;
  }
  return cachedI18n;
}

function tf(key, englishFallback) {
  const bundle = getI18nBundle();
  if (bundle && bundle.catalog) {
    const translated =
      lookupCatalogString(bundle.catalog, key) ||
      lookupCatalogString(bundle.fallbackCatalog, key);
    if (typeof translated === 'string') {
      return translated;
    }
  }
  return englishFallback;
}

function toStructuredRendererError(result, fallbackMessage) {
  const info = result && result.errorInfo && typeof result.errorInfo === 'object'
    ? result.errorInfo
    : null;
  const message =
    (info && typeof info.message === 'string' && info.message) ||
    (result && typeof result.error === 'string' && result.error) ||
    fallbackMessage ||
    tf('preload.unknownError', 'Unknown error');
  const error = new Error(message);
  if (info && typeof info.code === 'string') {
    error.code = info.code;
    if (info.code === 'E_CANCELED') {
      error.name = 'E_CANCELED';
    }
  }
  if (info && typeof info.timestamp === 'number') {
    error.timestamp = info.timestamp;
  }
  return error;
}

contextBridge.exposeInMainWorld('api', {
  /** Host OS: 'win32' | 'darwin' | 'linux' | ... */
  platform: process.platform,
  getI18n: () => {
    cachedI18n = ipcRenderer.sendSync('i18n:get');
    return cachedI18n;
  },
  getThemePreference: () => {
    try {
      const payload = ipcRenderer.sendSync('theme:get');
      if (payload && typeof payload.theme === 'string') {
        return payload.theme;
      }
    } catch {
      /* fall through */
    }
    return 'system';
  },
  getPathForFile: (file) => {
    try {
      return webUtils.getPathForFile(file);
    } catch {
      return '';
    }
  },
  sendMessage: async (text, options = {}) => {
    const enableThinking = options.enableThinking === true;
    const resubmit = options.resubmit !== false;
    const files = Array.isArray(options.files) ? options.files : [];
    const result = await ipcRenderer.invoke('engine:sendMessage', {
      message: text,
      enableThinking,
      resubmit,
      files,
    });
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.unknownInferenceError', 'Unknown inference error'));
    }
    return result.response;
  },
  sendMessageStream: (text, handlers = {}) => {
    const enableThinking = handlers.enableThinking === true;
    const resubmit = handlers.resubmit !== false;
    const files = Array.isArray(handlers.files) ? handlers.files : [];
    const requestId = `stream-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    const onToken = typeof handlers.onToken === 'function' ? handlers.onToken : null;
    const onSnapshot = typeof handlers.onSnapshot === 'function' ? handlers.onSnapshot : null;
    const onStarted = typeof handlers.onStarted === 'function' ? handlers.onStarted : null;
    const onDone = typeof handlers.onDone === 'function' ? handlers.onDone : null;
    const onError = typeof handlers.onError === 'function' ? handlers.onError : null;

    let settled = false;
    let resolveCompletion;
    let rejectCompletion;
    const completion = new Promise((resolve, reject) => {
      resolveCompletion = resolve;
      rejectCompletion = reject;
    });

    const cleanup = () => {
      ipcRenderer.removeListener('engine:streamEvent', listener);
    };

    const listener = (_event, payload) => {
      if (!payload || payload.requestId !== requestId) {
        return;
      }
      if (payload.type === 'started') {
        if (onStarted) {
          onStarted({ startsInThinking: Boolean(payload.startsInThinking) });
        }
        return;
      }
      if (payload.type === 'chunk') {
        if (onToken) {
          onToken(payload.chunk || '');
        }
        return;
      }
      if (payload.type === 'snapshot') {
        if (onSnapshot) {
          onSnapshot(typeof payload.text === 'string' ? payload.text : '');
        }
        return;
      }
      if (payload.type === 'done') {
        if (settled) return;
        settled = true;
        cleanup();
        if (onDone) {
          onDone(payload.response || '');
        }
        resolveCompletion(payload.response || '');
        return;
      }
      if (payload.type === 'canceled') {
        if (settled) return;
        settled = true;
        cleanup();
        const error = new Error(tf('preload.generationCanceled', 'Generation canceled.'));
        error.name = 'E_CANCELED';
        error.code = 'E_CANCELED';
        if (onError) {
          onError(error);
        }
        rejectCompletion(error);
        return;
      }
      if (payload.type === 'error') {
        if (settled) return;
        settled = true;
        cleanup();
        const error = toStructuredRendererError({ errorInfo: payload.errorInfo }, tf('preload.unknownStreamError', 'Unknown stream error'));
        if (onError) {
          onError(error);
        }
        rejectCompletion(error);
      }
    };

    ipcRenderer.on('engine:streamEvent', listener);
    ipcRenderer.send('engine:streamStart', { requestId, message: text, enableThinking, resubmit, files });

    return {
      requestId,
      completion,
      cancel: () => {
        if (settled) return;
        ipcRenderer.send('engine:streamCancel', { requestId });
        void ipcRenderer.invoke('engine:cancel');
      },
    };
  },
  getStatus: async () => {
    const result = await ipcRenderer.invoke('engine:getStatus');
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.unknownEngineError', 'Unknown engine error'));
    }
    return result.status;
  },
  getContextUsage: async (resubmit = true, options = {}) => {
    const result = await ipcRenderer.invoke('engine:getContextUsage', {
      resubmit: resubmit !== false,
      refresh: Boolean(options && options.refresh),
    });
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotReadContextUsage', 'Could not read context usage'));
    }
    return result.usage;
  },
  bootstrapEngine: async () => {
    const result = await ipcRenderer.invoke('engine:bootstrap');
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.engineFailedToStart', 'Engine failed to start'));
    }
    return {
      modelId: result.modelId ?? null,
      fellBack: Boolean(result.fellBack),
      clearedPreference: Boolean(result.clearedPreference),
      loadFailed: Boolean(result.loadFailed),
      message: typeof result.message === 'string' ? result.message : null,
      pending: Boolean(result.pending),
    };
  },
  getPreferences: async () => {
    const result = await ipcRenderer.invoke('prefs:get');
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotReadPreferences', 'Could not read preferences'));
    }
    return result.preferences;
  },
  updatePreferences: async (partial) => {
    const result = await ipcRenderer.invoke('prefs:update', partial || {});
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotSavePreferences', 'Could not save preferences'));
    }
    return result.preferences;
  },
  getSelectedModelId: async () => {
    const result = await ipcRenderer.invoke('prefs:getSelectedModelId');
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotReadSelectedModel', 'Could not read selected model'));
    }
    return result.modelId ?? null;
  },
  setSelectedModelId: async (modelId) => {
    const result = await ipcRenderer.invoke('prefs:setSelectedModelId', { modelId });
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotSaveSelectedModel', 'Could not save selected model'));
    }
    return result.modelId;
  },
  listCachedModels: async () => {
    const result = await ipcRenderer.invoke('prefs:listCachedModels');
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotListCachedModels', 'Could not list cached models'));
    }
    return result.models;
  },
  moveCachedModelToTrash: async (modelId) => {
    const result = await ipcRenderer.invoke('prefs:moveCachedModelToTrash', { modelId });
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotMoveModelToRecycleBin', 'Could not move model to the recycle bin'));
    }
    return {
      wasSelected: Boolean(result.wasSelected),
    };
  },
  cancelModelDownload: async (modelId) => {
    const result = await ipcRenderer.invoke('models:cancelModelDownload', { modelId });
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotStopDownload', 'Could not stop download'));
    }
    return { cancelled: Boolean(result.cancelled), modelId: result.modelId };
  },
  probeHubRepo: async (modelId) => {
    const result = await ipcRenderer.invoke('models:probeHubRepo', { modelId });
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotInspectHubRepository', 'Could not inspect Hub repository'));
    }
    return {
      modelId: result.modelId,
      kind: result.kind,
      variants: Array.isArray(result.variants) ? result.variants : [],
      files: Array.isArray(result.files) ? result.files : [],
    };
  },
  downloadModel: async (modelId, handlers = {}) => {
    const onProgress =
      handlers && typeof handlers.onProgress === 'function' ? handlers.onProgress : null;
    const allowPatterns =
      handlers && Array.isArray(handlers.allowPatterns) ? handlers.allowPatterns : undefined;
    const ggufVariant =
      handlers && typeof handlers.ggufVariant === 'string' ? handlers.ggufVariant : undefined;
    const channel = 'models:downloadProgress';
    const listener = (_event, payload) => {
      if (!payload || payload.modelId !== modelId || !onProgress) {
        return;
      }
      onProgress(payload.event);
    };
    if (onProgress) {
      ipcRenderer.on(channel, listener);
    }
    try {
      const result = await ipcRenderer.invoke('models:downloadModel', {
        modelId,
        allowPatterns,
        ggufVariant,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.modelDownloadFailed', 'Model download failed'));
      }
      return result.modelId;
    } finally {
      if (onProgress) {
        ipcRenderer.removeListener(channel, listener);
      }
    }
  },
  reinitializeEngine: async () => {
    const result = await ipcRenderer.invoke('engine:reinitialize');
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotReloadModel', 'Could not reload model'));
    }
    return {
      modelId: result.modelId ?? null,
      fellBack: Boolean(result.fellBack),
      clearedPreference: Boolean(result.clearedPreference),
      loadFailed: Boolean(result.loadFailed),
    };
  },
  ejectEngine: async () => {
    const result = await ipcRenderer.invoke('engine:eject');
    if (!result.ok) {
      throw toStructuredRendererError(result, tf('preload.couldNotEjectModel', 'Could not eject model'));
    }
    return { modelId: result.modelId ?? null };
  },
  onInitProgress: (callback) => {
    const listener = (_event, data) => callback(data);
    ipcRenderer.on('engine:initProgress', listener);
    return () => ipcRenderer.removeListener('engine:initProgress', listener);
  },
  resources: {
    listTree: async () => {
      const result = await ipcRenderer.invoke('resources:listTree');
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return result.tree;
    },
    uploadFilesWithMode: async (sourcePaths, targetFolderPath, conflictMode, progressToken) => {
      const result = await ipcRenderer.invoke('resources:uploadFiles', {
        sourcePaths,
        targetFolderPath,
        conflictMode,
        progressToken,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return {
        tree: result.tree,
        stats: result.stats || null,
      };
    },
    uploadFileContentsWithMode: async (
      files,
      targetFolderPath,
      conflictMode,
      progressToken
    ) => {
      const result = await ipcRenderer.invoke('resources:uploadFileContents', {
        files,
        targetFolderPath,
        conflictMode,
        progressToken,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return {
        tree: result.tree,
        stats: result.stats || null,
      };
    },
    getUploadConflictsForPaths: async (sourcePaths, targetFolderPath) => {
      const result = await ipcRenderer.invoke('resources:getUploadConflictsForPaths', {
        sourcePaths,
        targetFolderPath,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return {
        conflictCount: result.conflictCount || 0,
        totalCount: result.totalCount || 0,
      };
    },
    getUploadConflictsForFileContents: async (files, targetFolderPath) => {
      const result = await ipcRenderer.invoke(
        'resources:getUploadConflictsForFileContents',
        {
          files,
          targetFolderPath,
        }
      );
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return {
        conflictCount: result.conflictCount || 0,
        totalCount: result.totalCount || 0,
      };
    },
    createFolder: async (parentFolderPath, folderName) => {
      const result = await ipcRenderer.invoke('resources:createFolder', {
        parentFolderPath,
        folderName,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return result.tree;
    },
    createFile: async (parentFolderPath, fileName) => {
      const result = await ipcRenderer.invoke('resources:createFile', {
        parentFolderPath,
        fileName,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return result.tree;
    },
    renameEntry: async (entryPath, newName, overwrite = false) => {
      const result = await ipcRenderer.invoke('resources:renameEntry', {
        entryPath,
        newName,
        overwrite,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return result.tree;
    },
    moveEntry: async (sourcePath, targetFolderPath, overwrite = false) => {
      const result = await ipcRenderer.invoke('resources:moveEntry', {
        sourcePath,
        targetFolderPath,
        overwrite,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return result.tree;
    },
    deleteEntry: async (entryPath) => {
      const result = await ipcRenderer.invoke('resources:deleteEntry', {
        entryPath,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return result.tree;
    },
    importFromOutputs: async (outputRelativePath, targetFolderPath) => {
      const result = await ipcRenderer.invoke('resources:importFromOutputs', {
        outputRelativePath,
        targetFolderPath,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownResourcesError', 'Unknown resources error'));
      }
      return result.tree;
    },
    onUploadProgress: (callback) => {
      if (typeof callback !== 'function') {
        return () => {};
      }
      const listener = (_event, payload) => {
        callback(payload);
      };
      ipcRenderer.on('resources:uploadProgress', listener);
      return () => {
        ipcRenderer.removeListener('resources:uploadProgress', listener);
      };
    },
  },
  outputs: {
    listTree: async () => {
      const result = await ipcRenderer.invoke('outputs:listTree');
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownOutputsError', 'Unknown outputs error'));
      }
      return result.tree;
    },
    createFolder: async (parentFolderPath, folderName) => {
      const result = await ipcRenderer.invoke('outputs:createFolder', {
        parentFolderPath,
        folderName,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownOutputsError', 'Unknown outputs error'));
      }
      return result.tree;
    },
    renameEntry: async (entryPath, newName, overwrite = false) => {
      const result = await ipcRenderer.invoke('outputs:renameEntry', {
        entryPath,
        newName,
        overwrite,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownOutputsError', 'Unknown outputs error'));
      }
      return result.tree;
    },
    moveEntry: async (sourcePath, targetFolderPath, overwrite = false) => {
      const result = await ipcRenderer.invoke('outputs:moveEntry', {
        sourcePath,
        targetFolderPath,
        overwrite,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownOutputsError', 'Unknown outputs error'));
      }
      return result.tree;
    },
    deleteEntry: async (entryPath) => {
      const result = await ipcRenderer.invoke('outputs:deleteEntry', {
        entryPath,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.unknownOutputsError', 'Unknown outputs error'));
      }
      return result.tree;
    },
    startNativeDrag: (relativePath) => {
      ipcRenderer.send('outputs:startNativeDrag', { relativePath });
    },
  },
  markdown: {
    openEditor: async (panel, relativePath) => {
      const result = await ipcRenderer.invoke('markdown:openEditor', {
        panel,
        relativePath,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotOpenMarkdownEditor', 'Could not open markdown editor'));
      }
    },
  },
  media: {
    openViewer: async (panel, relativePath) => {
      const result = await ipcRenderer.invoke('media:openViewer', {
        panel,
        relativePath,
      });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotOpenMediaViewer', 'Could not open media viewer'));
      }
    },
  },
  sessions: {
    list: async () => {
      const result = await ipcRenderer.invoke('sessions:list');
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotListSessions', 'Could not list sessions'));
      }
      return result.sessions || [];
    },
    getActive: async () => {
      const result = await ipcRenderer.invoke('sessions:getActive');
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotReadActiveSession', 'Could not read active session'));
      }
      return result.sessionName ?? null;
    },
    new: async () => {
      const result = await ipcRenderer.invoke('sessions:new');
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotStartNewSession', 'Could not start a new session'));
      }
    },
    load: async (sessionName) => {
      const result = await ipcRenderer.invoke('sessions:load', { sessionName });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotLoadSession', 'Could not load session'));
      }
      return {
        sessionName: result.sessionName,
        messages: result.messages || [],
      };
    },
    rename: async (oldName, newName) => {
      const result = await ipcRenderer.invoke('sessions:rename', { oldName, newName });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotRenameSession', 'Could not rename session'));
      }
      return {
        sessionName: result.sessionName,
        sessions: result.sessions || [],
      };
    },
    trash: async (sessionName) => {
      const result = await ipcRenderer.invoke('sessions:trash', { sessionName });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotDeleteSession', 'Could not delete session'));
      }
      return {
        sessionName: result.sessionName,
        wasActive: Boolean(result.wasActive),
        sessions: result.sessions || [],
      };
    },
    deleteMessage: async (messageIndex) => {
      const result = await ipcRenderer.invoke('sessions:deleteMessage', { messageIndex });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotDeleteMessage', 'Could not delete message'));
      }
      return result.messages || [];
    },
  },
  workspaces: {
    list: async () => {
      const result = await ipcRenderer.invoke('workspaces:list');
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotListWorkspaces', 'Could not list workspaces'));
      }
      return {
        workspaces: result.workspaces || [],
        active: result.active ?? null,
      };
    },
    getActive: async () => {
      const result = await ipcRenderer.invoke('workspaces:getActive');
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotReadActiveWorkspace', 'Could not read active workspace'));
      }
      return result.name ?? null;
    },
    setActive: async (name) => {
      const result = await ipcRenderer.invoke('workspaces:setActive', { name });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotSwitchWorkspace', 'Could not switch workspace'));
      }
      return {
        workspaces: result.workspaces || [],
        active: result.active ?? null,
      };
    },
    create: async () => {
      const result = await ipcRenderer.invoke('workspaces:create');
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotCreateWorkspace', 'Could not create workspace'));
      }
      return {
        name: result.name,
        workspaces: result.workspaces || [],
        active: result.active ?? null,
      };
    },
    rename: async (oldName, newName) => {
      const result = await ipcRenderer.invoke('workspaces:rename', { oldName, newName });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotRenameWorkspace', 'Could not rename workspace'));
      }
      return {
        name: result.name,
        workspaces: result.workspaces || [],
        active: result.active ?? null,
      };
    },
    trash: async (name) => {
      const result = await ipcRenderer.invoke('workspaces:trash', { name });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotDeleteWorkspace', 'Could not delete workspace'));
      }
      return {
        name: result.name,
        wasActive: Boolean(result.wasActive),
        workspaces: result.workspaces || [],
        active: result.active ?? null,
      };
    },
  },
  chat: {
    exportMessage: async (messageIndex) => {
      const result = await ipcRenderer.invoke('chat:exportMessage', { messageIndex });
      if (!result.ok) {
        throw toStructuredRendererError(result, tf('preload.couldNotExportMessage', 'Could not export message'));
      }
      return {
        tree: result.tree,
        fileName: result.fileName,
      };
    },
  },
});
