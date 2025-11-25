# zk_dtypes Style Guide

## Introduction

This document defines the coding standards for python code in zk_dtypes. The
base guideline is the [Google Python Style Guide], combined with the
[Angular Commit Convention], with explicit project-specific modifications. In
addition to code style, this guide incorporates our rules for commit messages,
pull requests, and IDE/editor setup.

______________________________________________________________________

## Core Principles

- **Readability:** Both code and commits should be immediately understandable.
- **Maintainability:** Code should be easy to refactor and extend.
- **Consistency:** Apply the same conventions across files and modules.
- **Performance:** Prioritize clarity, but optimize carefully where latency and
  cost are critical.

______________________________________________________________________

## Python Coding Style

The following are project-specific deviations and clarifications from the
[Google Python Style Guide].

______________________________________________________________________

## Comment Style

- Non-trivial code changes must be accompanied by comments.

- Comments explain **why** a change or design decision was made or explain the
  code for better readability.

- Use full sentences with proper punctuation.

- Do not use **double spaces** in comments. Always use a **single space** after
  periods.

  ```python
  # ✅ Correct: This is a proper comment. It follows the rule.
  # ❌ Wrong: This is an improper comment.  It has double spaces.
  ```

______________________________________________________________________

## Testing

- **Framework**: Use absl-py test.
- **Coverage**: New features must include tests whenever applicable.
- **Completeness**: Always include boundary cases and error paths.
- **Determinism**: Tests must be deterministic and runnable independently (no
  hidden state dependencies).
- **Performance**: Add benchmarks for performance-critical code paths when
  appropriate.

______________________________________________________________________

## Collaboration Rules

### Commits (Angular Commit Convention)

- Must follow the [Commit Message Guideline].

- Format:

  ```
  <type>(<scope>): <summary>
  ```

  where `type` ∈ {build, chore, ci, docs, feat, fix, perf, refactor, style,
  test}.

- Commit body: explain **why** the change was made (minimum 20 characters).

- Footer: record breaking changes, deprecations, and related issues/PRs.

- Each commit must include only **minimal, logically related changes**. Avoid
  mixing style fixes with functional changes.

### Pull Requests

- Follow the [Pull Request Guideline].
- Commits must be **atomic** and independently buildable/testable.
- Provide context and links (short SHA for external references).

### File Formatting

- Every file must end with a single newline.
- No trailing whitespace.
- No extra blank lines at EOF.

______________________________________________________________________

## Tooling

- **Formatter:** `pyink`.
- **Linter:** `pylint`. Refer to the [.pylintrc] file in the repo.
- **Pre-commit hooks:** Recommended for enforcing format and lint locally.
- **CI:** All PRs must pass lint, format, and tests before merge.

[.pylintrc]: /.pylintrc
[angular commit convention]: https://github.com/angular/angular/blob/main/contributing-docs/commit-message-guidelines.md
[commit message guideline]: https://github.com/fractalyze/.github/blob/main/COMMIT_MESSAGE_GUIDELINE.md
[google python style guide]: https://google.github.io/styleguide/pyguide.html
[pull request guideline]: https://github.com/fractalyze/.github/blob/main/PULL_REQUEST_GUIDELINE.md
