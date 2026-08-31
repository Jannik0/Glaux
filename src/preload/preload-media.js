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

contextBridge.exposeInMainWorld('mediaApi', {
  getContext: async () => {
    const result = await ipcRenderer.invoke('media:getContext');
    if (!result.ok) {
      throw toError(result, 'Could not read viewer context');
    }
    return result.context;
  },
});
