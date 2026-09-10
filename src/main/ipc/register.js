const { registerEngineBridgeIpc } = require('../domains/engineBridge');
const { registerModelsPrefsIpc } = require('../domains/modelsPrefs');
const { registerPreferencesIpc } = require('../domains/preferences');
const { registerI18nIpc } = require('../domains/i18n');
const { registerThemeIpc } = require('../domains/theme');
const { registerWorkspacesIpc } = require('../domains/workspaces');
const { registerSessionsIpc } = require('../domains/sessions');
const { registerResourcesIpc } = require('../domains/resources');
const { registerOutputsIpc } = require('../domains/outputs');
const { registerEditorsIpc } = require('../domains/editors');

function registerIpc() {
  registerI18nIpc();
  registerThemeIpc();
  registerPreferencesIpc();
  registerEngineBridgeIpc();
  registerModelsPrefsIpc();
  registerWorkspacesIpc();
  registerSessionsIpc();
  registerResourcesIpc();
  registerOutputsIpc();
  registerEditorsIpc();
}

module.exports = {
  registerIpc,
};
