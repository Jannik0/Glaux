const path = require('path');
const { ipcMain, nativeImage } = require('electron');
const { createWorkspaceFs } = require('./workspaceFs');
const {
  getOutputsRoot,
  resolveOutputsPath,
  APP_ICON_PNG_PATH,
} = require('../paths');

const outputsFs = createWorkspaceFs({
  getRoot: getOutputsRoot,
  rootName: 'outputs',
  uploadProgressChannel: 'resources:uploadProgress',
});

function registerOutputsIpc() {
  ipcMain.handle('outputs:listTree', async () => {
    try {
      const tree = await outputsFs.getTree();
      return { ok: true, tree };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });

  ipcMain.handle(
    'outputs:createFolder',
    async (_event, { parentFolderPath, folderName }) => {
      try {
        const tree = await outputsFs.createFolder(parentFolderPath, folderName);
        return { ok: true, tree };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle(
    'outputs:renameEntry',
    async (_event, { entryPath, newName, overwrite = false }) => {
      try {
        const tree = await outputsFs.renameEntry(entryPath, newName, overwrite);
        return { ok: true, tree };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle(
    'outputs:moveEntry',
    async (_event, { sourcePath, targetFolderPath, overwrite = false }) => {
      try {
        const tree = await outputsFs.moveEntry(sourcePath, targetFolderPath, overwrite);
        return { ok: true, tree };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle('outputs:deleteEntry', async (_event, { entryPath }) => {
    try {
      const tree = await outputsFs.deleteEntry(entryPath);
      return { ok: true, tree };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });

  ipcMain.on('outputs:startNativeDrag', (event, { relativePath }) => {
    try {
      const absolutePath = resolveOutputsPath(relativePath);
      event.sender.startDrag({
        file: absolutePath,
        icon: nativeImage.createFromPath(APP_ICON_PNG_PATH).resize({ width: 24, height: 24 }),
      });
    } catch (_err) {
      // Silently ignore drag failures.
    }
  });
}

module.exports = {
  registerOutputsIpc,
  outputsFs,
};
