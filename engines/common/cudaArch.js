'use strict';

/**
 * CUDA architectures shipped with Glaux. One list for ggml's compile and for
 * pruning prebuilt fatbins. `-real` cubins cover these compute capabilities;
 * 12.0 also keeps PTX so newer NVIDIA GPUs can JIT.
 */
const CUDA_CUBIN_ARCHITECTURES = Object.freeze([75, 80, 86, 89, 90, 100, 120]);
const CUDA_PTX_ARCHITECTURE = 120;

const cubinArchs = new Set(CUDA_CUBIN_ARCHITECTURES);

/**
 * CMake list: every arch except the forward PTX arch is marked `-real`.
 * @returns {string}
 */
function cudaGgmlArchitectureList() {
  return CUDA_CUBIN_ARCHITECTURES.map((arch) =>
    arch === CUDA_PTX_ARCHITECTURE ? String(arch) : `${arch}-real`
  ).join(';');
}

/**
 * @param {number} major
 * @param {number} minor
 * @returns {number | null}
 */
function capabilityKey(major, minor) {
  if (!Number.isInteger(major) || !Number.isInteger(minor) || major < 0 || minor < 0 || minor > 9) {
    return null;
  }
  return major * 10 + minor;
}

/**
 * True when this GPU can run a bundled cubin, or is newer than 12.0 and can
 * JIT the forward PTX. 10.3 and 8.7 are not covered by either.
 * @param {number} major
 * @param {number} minor
 * @returns {boolean}
 */
function cudaComputeCapabilitySupported(major, minor) {
  const key = capabilityKey(major, minor);
  if (key == null) {
    return false;
  }
  return cubinArchs.has(key) || key > CUDA_PTX_ARCHITECTURE;
}

/**
 * Whether a fatbin image should stay. Kind 1 is PTX, kind 2 is a cubin.
 * sm_90a / sm_100a share the base architecture number, so they stay with 9.0 / 10.0.
 * @param {number} kind
 * @param {number} arch
 * @returns {boolean}
 */
function keepCudaFatbinImage(kind, arch) {
  if (!Number.isInteger(arch) || arch < 0) {
    return true;
  }
  if (kind === 1) {
    return arch >= CUDA_PTX_ARCHITECTURE;
  }
  if (kind === 2) {
    return cubinArchs.has(arch);
  }
  if (arch === 0) {
    return true;
  }
  return cubinArchs.has(arch) || arch >= CUDA_PTX_ARCHITECTURE;
}

/**
 * @param {string} text
 * @returns {boolean}
 */
function isNoCudaKernelImage(text) {
  return /no kernel image is available for execution on the device|cudaErrorNoKernelImage/i.test(
    String(text || '')
  );
}

/**
 * Decide CUDA_VISIBLE_DEVICES for a probed GPU list.
 * `visible === undefined` means the variable is unset. Empty string and `-1`
 * already hide CUDA and are left alone. Non-index values (UUIDs) are left alone.
 *
 * @param {{ index: number, major: number, minor: number }[]} gpus
 * @param {string | undefined} visible
 * @returns {string | undefined} value to set, or undefined to leave the variable unchanged
 */
function selectCudaVisibleDevices(gpus, visible) {
  const list = Array.isArray(gpus) ? gpus : [];
  if (visible != null) {
    const raw = String(visible).trim();
    if (raw === '' || raw === '-1') {
      return undefined;
    }
    if (!/^\d+(,\d+)*$/.test(raw)) {
      return undefined;
    }
    const indexes = raw.split(',').map((part) => Number(part));
    const kept = indexes.filter((index) => {
      const gpu = list.find((item) => item.index === index);
      if (!gpu) {
        return true;
      }
      return cudaComputeCapabilitySupported(gpu.major, gpu.minor);
    });
    if (kept.length === indexes.length) {
      return undefined;
    }
    if (kept.length === 0) {
      return '-1';
    }
    return kept.join(',');
  }

  const kept = list.filter((gpu) => cudaComputeCapabilitySupported(gpu.major, gpu.minor));
  if (kept.length === list.length) {
    return undefined;
  }
  if (kept.length === 0) {
    return '-1';
  }
  return kept.map((gpu) => gpu.index).join(',');
}

module.exports = {
  CUDA_CUBIN_ARCHITECTURES,
  CUDA_PTX_ARCHITECTURE,
  cudaGgmlArchitectureList,
  cudaComputeCapabilitySupported,
  keepCudaFatbinImage,
  isNoCudaKernelImage,
  selectCudaVisibleDevices,
};
