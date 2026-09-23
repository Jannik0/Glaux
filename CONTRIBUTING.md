# Contributing to Glaux

Thanks for helping. Contributions of every size are welcome: bug reports, model/use-case testing (especially on **macOS**), docs, and code.

This file is about how to work on Glaux. Full product setup, GPU notes, and packaging live in [README.md](README.md). Private security reports go to [SECURITY.md](SECURITY.md) — do not open a public issue for vulnerabilities.


## Ways to help

- Test Hub models and real workflows; note OS, GPU, engine (safetensors / llama.cpp / transcribe.cpp), and model id.
- Reproduce and fix bugs. Include steps, OS, whether you used `npm start` or a packaged build, and any relevant logs.
- Improve docs, translations, or accessibility.
- Add features that stay local: Glaux is an on-device workspace, not a cloud API client.


## Planned features

These are on the roadmap. Please open an issue before starting a large implementation so the approach can be aligned:

- **Cross-session memory** — persist useful context across chats and sessions, not only within one conversation.
- **Web research** — let the model look up current information on the web when the user asks for it.
- **Tool usage** — let the model call tools (files, commands, and similar) as part of a turn.


## Development

Follow [Development setup](README.md#development-setup) in the README (`npm install`, Python or `npm run build:python`, optional native engines). Then:

```bash
npm start
npm test
```

Windows and Linux are tested platforms. macOS still needs validation. GPU backends are compiled into the vendor trees at build time; end users only need a driver.

Do **not** commit generated or huge trees: `vendor/`, `dist/`, `deps/`, `node_modules/`, `.env/`. Do not commit Hugging Face tokens, preferences from your user profile, or model weights.


## Pull requests

1. Keep the change focused. Unrelated refactors make review harder.
2. Match the style of the files you touch (Electron CommonJS, existing Python worker layout).
3. Run `npm test`. Add or extend a test in `tests/` when the change is covered by the Node unit suite (routing, path sandbox, GPU helpers, i18n, ffmpeg, engine args).
4. If you change UI strings, update **all** catalogs under `src/i18n/locales/` (`en`, `de`, `fr`, `es`, `it`, `pt`) so keys stay in sync. English is the fallback.
5. Describe **why** in the PR, not only what. Mention platforms you tried.

There is no CLA. By contributing you agree the work is licensed under the project [MIT license](LICENSE).


## Scope notes

- Inference stays on the user’s machine. Do not add telemetry or cloud inference as a default path.
- Packaged Windows NSIS installers and Linux DEB and RPM packages must stay under GitHub’s **2 GiB** per-file limit. Size-sensitive Python/native cuts belong in `build:python` / `build:ffmpeg` / `build:llamacpp` / `build:transcribe`, not extra shrink scripts or one-off copies in `dist/`.
- llama.cpp and transcribe.cpp are **pinned clones** in `deps/` (gitignored). Prefer Glaux-side engine bridges and build scripts over vendoring a full fork unless the change truly belongs upstream.

Questions about a change are fine as an issue before you invest in a large PR.
