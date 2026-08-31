# Security Policy

## Supported versions

Security fixes are applied on a best-effort basis to the latest source on the default branch and to the most recent published release when practical.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Use **GitHub private vulnerability reporting**: *Security* → *Report a vulnerability* on the repository.

Include as much detail as you can:

- A clear description of the issue and its impact
- Steps to reproduce, or a minimal proof of concept
- Affected version / commit, OS, and whether you used a packaged build or `npm start`
- Any suggested fix, if you have one

## What to expect

We will acknowledge reports when we can and work on a fix or mitigation for confirmed issues. Disclosure timelines depend on severity and complexity; please give us a reasonable window before public discussion.

## Scope notes

Glaux runs **local** inference and stores user data on disk. Reports involving local privilege escalation in the app, unsafe handling of untrusted model or media inputs, path traversal around the resource/output sandboxes, or supply-chain issues in packaging scripts are especially welcome. Issues that only affect third-party models or Hub content (without a Glaux bug) should be reported upstream where appropriate.
