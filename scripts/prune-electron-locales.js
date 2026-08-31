'use strict';

/**
 * afterPack: keep Chromium locale packs whose primary language Glaux ships,
 * delete the rest. electron-builder's electronLanguages matcher is inverted
 * (it does not treat "en" as matching en-US.pak), so this hook does the filter.
 */

const fs = require('fs/promises');
const path = require('path');
const { isSupportedElectronLocaleFile } = require('../src/i18n/languages');

/**
 * @param {import('app-builder-lib').AfterPackContext} context
 * @returns {string[]}
 */
function localeRoots(context) {
  const { appOutDir, electronPlatformName, packager } = context;
  if (electronPlatformName === 'darwin') {
    const appName = packager.appInfo.productFilename;
    const bundle = path.join(appOutDir, `${appName}.app`);
    return [
      path.join(bundle, 'Contents', 'Resources'),
      path.join(
        bundle,
        'Contents',
        'Frameworks',
        'Electron Framework.framework',
        'Resources'
      ),
      path.join(
        bundle,
        'Contents',
        'Frameworks',
        'Electron Framework.framework',
        'Versions',
        'A',
        'Resources'
      ),
    ];
  }
  return [path.join(appOutDir, 'locales')];
}

/**
 * @param {string} dir
 * @param {string} ext
 * @returns {Promise<number>}
 */
async function pruneDir(dir, ext) {
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return 0;
  }
  let removed = 0;
  for (const ent of entries) {
    if (path.extname(ent.name).toLowerCase() !== ext) {
      continue;
    }
    const basename = path.basename(ent.name, ext);
    if (isSupportedElectronLocaleFile(basename)) {
      continue;
    }
    await fs.rm(path.join(dir, ent.name), { recursive: true, force: true });
    removed += 1;
  }
  return removed;
}

/**
 * @param {import('app-builder-lib').AfterPackContext} context
 */
module.exports = async function pruneElectronLocales(context) {
  const ext = context.electronPlatformName === 'darwin' ? '.lproj' : '.pak';
  let removed = 0;
  for (const dir of localeRoots(context)) {
    removed += await pruneDir(dir, ext);
  }
  if (removed) {
    console.log(`Pruned ${removed} Electron locale pack(s) not used by Glaux`);
  }
};
