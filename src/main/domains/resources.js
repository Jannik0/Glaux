const path = require('path');
const fs = require('fs/promises');
const { ipcMain } = require('electron');
const { t } = require('../../i18n');
const { createWorkspaceFs } = require('./workspaceFs');
const {
  getResourcesRoot,
  getOutputsRoot,
  resolveOutputsPath,
  ensureResourcesDirectory,
  ensureOutputsDirectory,
} = require('../paths');

const resourcesFs = createWorkspaceFs({
  getRoot: getResourcesRoot,
  rootName: 'resources',
  uploadProgressChannel: 'resources:uploadProgress',
});

function registerResourcesIpc() {
  ipcMain.handle('resources:listTree', async () => {
    try {
      const tree = await resourcesFs.getTree();
      return { ok: true, tree };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });

  ipcMain.handle(
    'resources:uploadFiles',
    async (event, { sourcePaths, targetFolderPath, overwrite = false, conflictMode, progressToken }) => {
      try {
        await ensureResourcesDirectory();
        const mode = conflictMode || (overwrite ? 'overwrite' : 'error');
        const tasks = await resourcesFs.buildSourcePathTasks(sourcePaths, targetFolderPath || '');
        const uploadCount = await resourcesFs.countTasksToUpload(tasks, mode);
        const progressEmitter = resourcesFs.createProgressEmitter(event.sender, progressToken, uploadCount);
        const stats = await resourcesFs.runCopyTasks(tasks, mode, progressEmitter);
        const tree = await resourcesFs.getTree();
        return { ok: true, tree, stats };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle(
    'resources:uploadFileContents',
    async (event, { files, targetFolderPath, overwrite = false, conflictMode, progressToken }) => {
      try {
        await ensureResourcesDirectory();
        const mode = conflictMode || (overwrite ? 'overwrite' : 'error');
        const tasks = await resourcesFs.buildUploadFileTasks(files, targetFolderPath || '');
        const uploadCount = await resourcesFs.countTasksToUpload(tasks, mode);
        const progressEmitter = resourcesFs.createProgressEmitter(event.sender, progressToken, uploadCount);
        const stats = await resourcesFs.runWriteTasks(tasks, mode, progressEmitter);
        const tree = await resourcesFs.getTree();
        return { ok: true, tree, stats };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle(
    'resources:getUploadConflictsForPaths',
    async (_event, { sourcePaths, targetFolderPath }) => {
      try {
        await ensureResourcesDirectory();
        const tasks = await resourcesFs.buildSourcePathTasks(sourcePaths, targetFolderPath || '');
        const conflictCount = await resourcesFs.countExistingTaskTargets(tasks);
        return { ok: true, conflictCount, totalCount: tasks.length };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle(
    'resources:getUploadConflictsForFileContents',
    async (_event, { files, targetFolderPath }) => {
      try {
        await ensureResourcesDirectory();
        const tasks = await resourcesFs.buildUploadFileTasks(files, targetFolderPath || '');
        const conflictCount = await resourcesFs.countExistingTaskTargets(tasks);
        return { ok: true, conflictCount, totalCount: tasks.length };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle(
    'resources:createFolder',
    async (_event, { parentFolderPath, folderName }) => {
      try {
        const tree = await resourcesFs.createFolder(parentFolderPath, folderName);
        return { ok: true, tree };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle(
    'resources:createFile',
    async (_event, { parentFolderPath, fileName }) => {
      try {
        const tree = await resourcesFs.createFile(parentFolderPath, fileName);
        return { ok: true, tree };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle(
    'resources:renameEntry',
    async (_event, { entryPath, newName, overwrite = false }) => {
      try {
        const tree = await resourcesFs.renameEntry(entryPath, newName, overwrite);
        return { ok: true, tree };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle(
    'resources:moveEntry',
    async (_event, { sourcePath, targetFolderPath, overwrite = false }) => {
      try {
        const tree = await resourcesFs.moveEntry(sourcePath, targetFolderPath, overwrite);
        return { ok: true, tree };
      } catch (err) {
        return { ok: false, error: err.message || String(err) };
      }
    }
  );

  ipcMain.handle('resources:deleteEntry', async (_event, { entryPath }) => {
    try {
      const tree = await resourcesFs.deleteEntry(entryPath);
      return { ok: true, tree };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });

  ipcMain.handle('resources:importFromOutputs', async (_event, { outputRelativePath, targetFolderPath }) => {
    try {
      await ensureResourcesDirectory();
      await ensureOutputsDirectory();
      if (typeof outputRelativePath !== 'string' || !outputRelativePath.trim()) {
        throw new Error(t('errors.resources.outputPathRequired'));
      }
      const sourcePath = resolveOutputsPath(outputRelativePath);
      const targetDir = resourcesFs.resolvePath(targetFolderPath || '');
      const fileName = path.basename(sourcePath);
      const destPath = path.join(targetDir, fileName);

      const destRelative = path.relative(getResourcesRoot(), destPath);
      if (
        destRelative === '..' ||
        destRelative.startsWith(`..${path.sep}`) ||
        path.isAbsolute(destRelative)
      ) {
        throw new Error(t('errors.resources.destinationEscapesResources'));
      }

      await fs.cp(sourcePath, destPath, { recursive: true });
      const tree = await resourcesFs.getTree();
      return { ok: true, tree };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  });
}

module.exports = {
  registerResourcesIpc,
  resourcesFs,
};
