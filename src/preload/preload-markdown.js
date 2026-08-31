const { contextBridge, ipcRenderer } = require('electron');

function toError(result, fallbackMessage) {
  const message =
    (result && typeof result.error === 'string' && result.error) ||
    fallbackMessage ||
    'Unknown error';
  return new Error(message);
}

contextBridge.exposeInMainWorld('api', {
  getI18n: () => ipcRenderer.sendSync('i18n:get'),
});

contextBridge.exposeInMainWorld('markdownApi', {
  getContext: async () => {
    const result = await ipcRenderer.invoke('markdown:getContext');
    if (!result.ok) {
      throw toError(result, 'Could not read editor context');
    }
    return result.context;
  },
  readFile: async () => {
    const result = await ipcRenderer.invoke('markdown:readFile');
    if (!result.ok) {
      throw toError(result, 'Could not read file');
    }
    return {
      content: result.content ?? '',
      fileName: result.fileName ?? '',
    };
  },
  writeFile: async (content) => {
    const result = await ipcRenderer.invoke('markdown:writeFile', { content });
    if (!result.ok) {
      throw toError(result, 'Could not save file');
    }
  },
  confirmClose: () => {
    ipcRenderer.send('markdown:confirmClose');
  },
});

ipcRenderer.on('markdown:attemptClose', () => {
  window.dispatchEvent(new CustomEvent('markdown-attempt-close'));
});
