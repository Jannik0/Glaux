'use strict';

const { describe, it, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const engineManager = require('../engines/engineManager');
const state = require('../src/main/state');
const {
  cancelPanelModelDownload,
  setModelDownloadCancelWaitMs,
  setModelCacheTrashOverride,
  MODEL_DOWNLOAD_CANCEL_WAIT_MS,
} = require('../src/main/domains/modelsPrefs');

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe('models panel download cancel ordering', () => {
  const original = {
    cancelModelDownload: engineManager.cancelModelDownload,
    resetInferenceWorker: engineManager.resetInferenceWorker,
  };

  beforeEach(() => {
    state.activePanelModelDownload = null;
    state.engineBootstrapped = false;
    state.selectedModelId = null;
    setModelDownloadCancelWaitMs(500);
    setModelCacheTrashOverride(null);
  });

  afterEach(() => {
    engineManager.cancelModelDownload = original.cancelModelDownload;
    engineManager.resetInferenceWorker = original.resetInferenceWorker;
    setModelCacheTrashOverride(null);
    setModelDownloadCancelWaitMs(MODEL_DOWNLOAD_CANCEL_WAIT_MS);
    state.activePanelModelDownload = null;
    state.engineBootstrapped = false;
    state.selectedModelId = null;
  });

  it('keeps the production cancel wait at 8 seconds', () => {
    assert.equal(MODEL_DOWNLOAD_CANCEL_WAIT_MS, 8000);
    setModelDownloadCancelWaitMs(40);
    assert.equal(MODEL_DOWNLOAD_CANCEL_WAIT_MS, 8000);
  });

  it('trashes the cache folder only after the download promise settles', async () => {
    const order = [];
    let releaseDownload;
    const download = new Promise((resolve) => {
      releaseDownload = resolve;
    });
    state.activePanelModelDownload = {
      modelId: 'org/settle-first',
      promise: download,
      loaded: 1,
      total: 4,
    };
    engineManager.cancelModelDownload = async () => {
      order.push('cancel');
    };
    engineManager.resetInferenceWorker = async () => {
      order.push('reset');
    };
    setModelCacheTrashOverride(async () => {
      order.push('trash');
    });

    const pending = cancelPanelModelDownload('org/settle-first');
    await delay(30);
    assert.deepEqual(order, ['cancel']);
    releaseDownload();
    assert.equal(await pending, 'org/settle-first');
    assert.deepEqual(order, ['cancel', 'trash']);
  });

  it('trashes the cache folder only after the wait and a worker reset', async () => {
    setModelDownloadCancelWaitMs(40);
    const order = [];
    state.activePanelModelDownload = {
      modelId: 'org/reset-first',
      promise: new Promise(() => {}),
      loaded: 0,
      total: 0,
    };
    engineManager.cancelModelDownload = async () => {
      order.push('cancel');
    };
    engineManager.resetInferenceWorker = async () => {
      order.push('reset');
    };
    setModelCacheTrashOverride(async () => {
      order.push('trash');
    });

    const pending = cancelPanelModelDownload('org/reset-first');
    await delay(15);
    assert.deepEqual(order, ['cancel']);
    assert.equal(await pending, 'org/reset-first');
    assert.deepEqual(order, ['cancel', 'reset', 'trash']);
    assert.equal(state.activePanelModelDownload, null);
  });
});
