'use strict';

/**
 * afterPack (invoked by `npm run dist`): Chromium locale filter, then restore
 * ELF SONAME links, the shared ggml CUDA backend, and shared CUDA 13 overlap
 * that electron-builder may have copied as full files. All size cuts already
 * ran in build:python / ffmpeg / llamacpp / transcribe — this hook does not
 * prune a second policy.
 */

const fs = require('fs');
const path = require('path');
const pruneElectronLocales = require('./prune-electron-locales');
const {
  collapseDuplicateLibsRecursive,
  finishStagedNativeDir,
  shareGgmlCudaBackend,
  shareTorchCuda13WithVendor,
} = require('./gpuBackends');

/**
 * @param {import('app-builder-lib').AfterPackContext} context
 * @returns {string}
 */
function resourcesDir(context) {
  const { appOutDir, electronPlatformName, packager } = context;
  if (electronPlatformName === 'darwin') {
    const appName = packager.appInfo.productFilename;
    return path.join(appOutDir, `${appName}.app`, 'Contents', 'Resources');
  }
  return path.join(appOutDir, 'resources');
}

/**
 * @param {import('app-builder-lib').AfterPackContext} context
 */
module.exports = async function afterPack(context) {
  await pruneElectronLocales(context);

  const resources = resourcesDir(context);
  if (!fs.existsSync(resources)) {
    return;
  }

  for (const name of ['cuda', 'ffmpeg', 'llamacpp', 'transcribe']) {
    finishStagedNativeDir(path.join(resources, name), { strip: false });
  }
  if (context.electronPlatformName !== 'darwin') {
    shareGgmlCudaBackend(path.join(resources, 'llamacpp'), path.join(resources, 'transcribe'));
  }

  const python = path.join(resources, 'python');
  const cuda = path.join(resources, 'cuda');
  if (!fs.existsSync(python)) {
    return;
  }

  const collapsed = collapseDuplicateLibsRecursive(python);
  if (collapsed) {
    console.log(`Collapsed ${collapsed} duplicate SONAME copy(ies) in python`);
  }
  if (fs.existsSync(cuda) && context.electronPlatformName !== 'darwin') {
    shareTorchCuda13WithVendor(python, cuda);
  }
};
