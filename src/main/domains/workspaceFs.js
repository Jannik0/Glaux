const path = require('path');
const fs = require('fs/promises');
const { t, appError } = require('../../i18n');
const { assertValidEntryName, toPosixRelative, isSubPath } = require('../pathSandbox');
const { pathExists, moveEntryToRecycleBin } = require('../paths');

/**
 * @param {{ getRoot: () => string, rootName: string, uploadProgressChannel: string }} config
 */
function createWorkspaceFs({ getRoot, rootName, uploadProgressChannel }) {
  function resolvePath(relativePath = '') {
    const root = getRoot();
    if (typeof relativePath !== 'string') {
      throw new Error(t('errors.workspaceFs.pathMustBeString'));
    }
    if (path.isAbsolute(relativePath)) {
      throw new Error(t('errors.workspaceFs.absolutePathsNotAllowed'));
    }
    const resolvedPath = path.resolve(root, relativePath);
    const relativeToRoot = path.relative(root, resolvedPath);
    if (
      relativeToRoot === '..' ||
      relativeToRoot.startsWith(`..${path.sep}`) ||
      path.isAbsolute(relativeToRoot)
    ) {
      throw new Error(t('errors.workspaceFs.pathEscapesDirectory', { rootName }));
    }
    return resolvedPath;
  }

  async function ensureDirectory() {
    await fs.mkdir(getRoot(), { recursive: true });
  }

  async function buildTree(absPath, relativePath = '', treeRootName = 'root') {
    const entries = await fs.readdir(absPath, { withFileTypes: true });
    const sorted = [...entries].sort((a, b) => {
      if (a.isDirectory() && !b.isDirectory()) return -1;
      if (!a.isDirectory() && b.isDirectory()) return 1;
      return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
    });

    const children = [];

    for (const entry of sorted) {
      const childRelativePath = toPosixRelative(
        relativePath ? path.join(relativePath, entry.name) : entry.name
      );
      const childAbsolutePath = path.join(absPath, entry.name);

      if (entry.isDirectory()) {
        const subtree = await buildTree(childAbsolutePath, childRelativePath);
        children.push(subtree);
        continue;
      }

      if (entry.isFile()) {
        children.push({
          type: 'file',
          name: entry.name,
          relativePath: childRelativePath,
        });
      }
    }

    return {
      type: 'folder',
      name: relativePath ? path.basename(relativePath) : treeRootName,
      relativePath: toPosixRelative(relativePath),
      children,
    };
  }

  async function getTree() {
    await ensureDirectory();
    return buildTree(getRoot(), '', rootName);
  }

  async function collectSourceCopyTasks(sourcePath, destinationPath, tasks) {
    const stats = await fs.stat(sourcePath).catch(() => null);
    if (!stats || (!stats.isFile() && !stats.isDirectory())) {
      throw new Error(t('errors.workspaceFs.sourceNotFileOrFolder', { sourcePath }));
    }

    if (stats.isDirectory()) {
      tasks.push({ type: 'dir', sourcePath, destinationPath });
      const entries = await fs.readdir(sourcePath, { withFileTypes: true });
      for (const entry of entries) {
        const childSource = path.join(sourcePath, entry.name);
        const childDestination = path.join(destinationPath, entry.name);
        if (entry.isDirectory() || entry.isFile()) {
          await collectSourceCopyTasks(childSource, childDestination, tasks);
        }
      }
      return;
    }

    tasks.push({ type: 'file', sourcePath, destinationPath });
  }

  async function buildSourcePathTasks(sourcePaths, targetFolderPath) {
    if (!Array.isArray(sourcePaths) || sourcePaths.length === 0) {
      throw new Error(t('errors.workspaceFs.noFilesOrFoldersProvided'));
    }

    const targetAbsolutePath = resolvePath(targetFolderPath || '');
    const targetStats = await fs.stat(targetAbsolutePath).catch(() => null);
    if (!targetStats || !targetStats.isDirectory()) {
      throw new Error(t('errors.workspaceFs.uploadTargetMissing'));
    }

    const tasks = [];
    for (const sourcePath of sourcePaths) {
      if (typeof sourcePath !== 'string' || !sourcePath) {
        throw new Error(t('errors.workspaceFs.invalidSourcePath'));
      }

      const sourceName = path.basename(path.normalize(sourcePath));
      if (!sourceName) {
        throw new Error(t('errors.workspaceFs.couldNotResolveSourceName', { sourcePath }));
      }

      const destinationPath = path.join(targetAbsolutePath, sourceName);
      await collectSourceCopyTasks(sourcePath, destinationPath, tasks);
    }

    return tasks;
  }

  function createProgressEmitter(sender, token, total) {
    let completed = 0;
    return {
      advance(current) {
        completed += 1;
        if (sender && token) {
          sender.send(uploadProgressChannel, {
            token,
            completed,
            total,
            current: current || '',
          });
        }
      },
    };
  }

  async function countTasksToUpload(tasks, conflictMode) {
    let uploadCount = 0;

    for (const task of tasks) {
      const destinationExists = await pathExists(task.destinationPath);

      if (task.type === 'dir') {
        if (!destinationExists) {
          uploadCount += 1;
          continue;
        }

        const destinationStats = await fs.stat(task.destinationPath).catch(() => null);
        if (!destinationStats) {
          continue;
        }

        if (destinationStats.isDirectory()) {
          if (conflictMode === 'error') {
            uploadCount += 1;
          }
          continue;
        }

        if (conflictMode === 'skip') {
          continue;
        }

        uploadCount += 1;
        continue;
      }

      if (!destinationExists || conflictMode !== 'skip') {
        uploadCount += 1;
      }
    }

    return uploadCount;
  }

  async function countExistingTaskTargets(tasks) {
    let existingCount = 0;
    for (const task of tasks) {
      if (await pathExists(task.destinationPath)) {
        existingCount += 1;
      }
    }
    return existingCount;
  }

  async function runCopyTasks(tasks, conflictMode, progressEmitter) {
    const root = getRoot();
    const stats = {
      totalCount: tasks.length,
      uploadedCount: 0,
      skippedCount: 0,
    };

    for (const task of tasks) {
      const destinationExists = await pathExists(task.destinationPath);
      const displayPath = toPosixRelative(path.relative(root, task.destinationPath));

      if (task.type === 'dir') {
        if (destinationExists) {
          const destinationStats = await fs.stat(task.destinationPath).catch(() => null);
          if (!destinationStats) {
            throw new Error(t('errors.workspaceFs.targetNotAccessible', { path: displayPath }));
          }

          if (destinationStats.isDirectory()) {
            if (conflictMode === 'error') {
              throw appError('errors.workspaceFs.targetAlreadyContains', { path: displayPath }, 'E_ALREADY_EXISTS');
            }
            stats.skippedCount += 1;
            continue;
          }

          if (conflictMode === 'skip') {
            stats.skippedCount += 1;
            continue;
          }

          await fs.rm(task.destinationPath, { recursive: true, force: true });
        }

        await fs.mkdir(task.destinationPath, { recursive: true });
        stats.uploadedCount += 1;
        progressEmitter.advance(displayPath);
        continue;
      }

      if (destinationExists) {
        if (conflictMode === 'error') {
          throw appError('errors.workspaceFs.targetAlreadyContains', { path: displayPath }, 'E_ALREADY_EXISTS');
        }
        if (conflictMode === 'skip') {
          stats.skippedCount += 1;
          continue;
        }

        await fs.rm(task.destinationPath, { recursive: true, force: true });
      }

      await fs.mkdir(path.dirname(task.destinationPath), { recursive: true });
      await fs.copyFile(task.sourcePath, task.destinationPath);
      stats.uploadedCount += 1;
      progressEmitter.advance(displayPath);
    }

    return stats;
  }

  async function buildUploadFileTasks(uploadFiles, targetFolderPath) {
    if (!Array.isArray(uploadFiles) || uploadFiles.length === 0) {
      throw new Error(t('errors.workspaceFs.noFilesProvided'));
    }

    const targetAbsolutePath = resolvePath(targetFolderPath || '');
    const targetStats = await fs.stat(targetAbsolutePath).catch(() => null);
    if (!targetStats || !targetStats.isDirectory()) {
      throw new Error(t('errors.workspaceFs.uploadTargetMissing'));
    }

    const tasks = [];
    for (const uploadFile of uploadFiles) {
      if (!uploadFile || typeof uploadFile !== 'object') {
        throw new Error(t('errors.workspaceFs.invalidUploadedFilePayload'));
      }

      const rawRelativePath =
        typeof uploadFile.relativePath === 'string' && uploadFile.relativePath.trim()
          ? uploadFile.relativePath
          : uploadFile.name;
      if (typeof rawRelativePath !== 'string' || !rawRelativePath.trim()) {
        throw new Error(t('errors.workspaceFs.uploadedFilePathMissing'));
      }

      const normalizedRelativePath = rawRelativePath
        .replace(/\\/g, '/')
        .replace(/^\/+/, '');
      const pathParts = normalizedRelativePath.split('/').filter(Boolean);
      if (!pathParts.length) {
        throw new Error(t('errors.workspaceFs.uploadedFilePathInvalid'));
      }

      const safeParts = pathParts.map((part) => assertValidEntryName(part));
      const destinationPath = path.join(targetAbsolutePath, ...safeParts);
      const content =
        uploadFile.content instanceof Uint8Array
          ? uploadFile.content
          : Uint8Array.from(uploadFile.content || []);

      tasks.push({
        type: 'file-content',
        destinationPath,
        relativePath: safeParts.join('/'),
        content,
      });
    }

    return tasks;
  }

  async function runWriteTasks(tasks, conflictMode, progressEmitter) {
    const stats = {
      totalCount: tasks.length,
      uploadedCount: 0,
      skippedCount: 0,
    };

    for (const task of tasks) {
      const destinationExists = await pathExists(task.destinationPath);
      const displayPath = task.relativePath;

      if (destinationExists) {
        if (conflictMode === 'error') {
          throw appError('errors.workspaceFs.targetAlreadyContains', { path: displayPath }, 'E_ALREADY_EXISTS');
        }
        if (conflictMode === 'skip') {
          stats.skippedCount += 1;
          continue;
        }
        await fs.rm(task.destinationPath, { recursive: true, force: true });
      }

      await fs.mkdir(path.dirname(task.destinationPath), { recursive: true });
      await fs.writeFile(task.destinationPath, Buffer.from(task.content));
      stats.uploadedCount += 1;
      progressEmitter.advance(displayPath);
    }

    return stats;
  }

  async function createFolder(parentFolderPath, folderName) {
    await ensureDirectory();
    const safeFolderName = assertValidEntryName(folderName);
    const parentAbsolutePath = resolvePath(parentFolderPath || '');
    const parentStats = await fs.stat(parentAbsolutePath).catch(() => null);

    if (!parentStats || !parentStats.isDirectory()) {
      throw new Error(t('errors.workspaceFs.parentFolderMissing'));
    }

    const nextFolderPath = path.join(parentAbsolutePath, safeFolderName);
    if (await pathExists(nextFolderPath)) {
      throw appError('errors.workspaceFs.alreadyExists', { name: safeFolderName }, 'E_ALREADY_EXISTS');
    }

    await fs.mkdir(nextFolderPath);
    return getTree();
  }

  async function createFile(parentFolderPath, fileName) {
    await ensureDirectory();
    const safeFileName = assertValidEntryName(fileName);
    const parentAbsolutePath = resolvePath(parentFolderPath || '');
    const parentStats = await fs.stat(parentAbsolutePath).catch(() => null);

    if (!parentStats || !parentStats.isDirectory()) {
      throw new Error(t('errors.workspaceFs.parentFolderMissing'));
    }

    const nextFilePath = path.join(parentAbsolutePath, safeFileName);
    if (await pathExists(nextFilePath)) {
      throw appError('errors.workspaceFs.alreadyExists', { name: safeFileName }, 'E_ALREADY_EXISTS');
    }

    await fs.writeFile(nextFilePath, '', 'utf8');
    return getTree();
  }

  async function renameEntry(entryPath, newName, overwrite = false) {
    await ensureDirectory();
    const safeName = assertValidEntryName(newName);
    const sourceAbsolutePath = resolvePath(entryPath || '');
    const sourceStats = await fs.stat(sourceAbsolutePath).catch(() => null);

    if (!sourceStats) {
      throw new Error(t('errors.workspaceFs.entryDoesNotExist'));
    }

    const parentPath = path.dirname(sourceAbsolutePath);
    const destinationPath = path.join(parentPath, safeName);

    if (destinationPath === sourceAbsolutePath) {
      return getTree();
    }

    if (await pathExists(destinationPath)) {
      if (!overwrite) {
        throw appError('errors.workspaceFs.alreadyExistsInFolder', { name: safeName }, 'E_ALREADY_EXISTS');
      }

      const destinationStats = await fs.stat(destinationPath).catch(() => null);
      if (!destinationStats) {
        throw new Error(t('errors.workspaceFs.renameDestinationUnavailable'));
      }

      if (!sourceStats.isFile() || !destinationStats.isFile()) {
        throw new Error(t('errors.workspaceFs.onlyFilesOverwriteRename'));
      }

      await fs.unlink(destinationPath);
    }

    await fs.rename(sourceAbsolutePath, destinationPath);
    return getTree();
  }

  async function moveEntry(sourcePath, targetFolderPath, overwrite = false) {
    await ensureDirectory();
    const sourceAbsolutePath = resolvePath(sourcePath || '');
    const sourceStats = await fs.stat(sourceAbsolutePath).catch(() => null);
    if (!sourceStats) {
      throw new Error(t('errors.workspaceFs.sourceEntryMissing'));
    }

    const targetAbsolutePath = resolvePath(targetFolderPath || '');
    const targetStats = await fs.stat(targetAbsolutePath).catch(() => null);
    if (!targetStats || !targetStats.isDirectory()) {
      throw new Error(t('errors.workspaceFs.targetFolderMissing'));
    }

    if (sourceStats.isDirectory() && isSubPath(sourceAbsolutePath, targetAbsolutePath)) {
      throw new Error(t('errors.workspaceFs.cannotMoveFolderIntoItself'));
    }

    const destinationPath = path.join(
      targetAbsolutePath,
      path.basename(sourceAbsolutePath)
    );

    if (destinationPath === sourceAbsolutePath) {
      return getTree();
    }

    if (await pathExists(destinationPath)) {
      if (!overwrite) {
        throw appError('errors.workspaceFs.targetContainsSameName', undefined, 'E_ALREADY_EXISTS');
      }

      const destinationStats = await fs.stat(destinationPath).catch(() => null);
      if (!destinationStats) {
        throw new Error(t('errors.workspaceFs.moveDestinationUnavailable'));
      }

      if (!sourceStats.isFile() || !destinationStats.isFile()) {
        throw new Error(t('errors.workspaceFs.onlyFilesOverwriteMove'));
      }

      await fs.unlink(destinationPath);
    }

    await fs.rename(sourceAbsolutePath, destinationPath);
    return getTree();
  }

  async function deleteEntry(entryPath) {
    await ensureDirectory();
    if (typeof entryPath !== 'string' || !entryPath.trim()) {
      throw new Error(t('errors.workspaceFs.entryPathRequired'));
    }

    const root = getRoot();
    const sourceAbsolutePath = resolvePath(entryPath);
    const rootRelative = path.relative(root, sourceAbsolutePath);
    if (!rootRelative) {
      throw new Error(t('errors.workspaceFs.cannotDeleteRoot', { rootName }));
    }

    await moveEntryToRecycleBin(sourceAbsolutePath);
    return getTree();
  }

  return {
    resolvePath,
    ensureDirectory,
    getTree,
    buildSourcePathTasks,
    buildUploadFileTasks,
    createProgressEmitter,
    countTasksToUpload,
    countExistingTaskTargets,
    runCopyTasks,
    runWriteTasks,
    createFolder,
    createFile,
    renameEntry,
    moveEntry,
    deleteEntry,
  };
}

module.exports = {
  createWorkspaceFs,
};
