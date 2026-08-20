# Local development rules

These rules apply to work under `tools/synthetic_wave_for_reg_baseline/`.

## Code and output organization

- Always use BART's GPU option (`-g`) for every reconstruction. Do not silently
  fall back to CPU. If a required BART command does not support `-g`, stop and
  document that command-specific exception before proceeding.
- Before adding code or generated outputs, choose a clear location within the
  existing structure. Keep reusable implementation in `scripts/`, focused tests
  in `tests/`, documentation in `docs/`, and task-specific requirements in
  `requirements/`. Do not place large generated artifacts in the source tree.
- Put generated experiment artifacts in a separate, descriptively named dataset
  output tree. Never overwrite accepted or historical results. Keep each run
  discoverable through a manifest or index that records its purpose, status,
  inputs, configuration, provenance, and canonical outputs.
- Prefer a small number of predictable entry points and shallow, navigable
  directories. Use stable descriptive names, avoid unexplained abbreviations,
  and update the nearest README, manifest, or output index when adding something
  that future users need to find.
- Preserve accessibility: provide dataset-independent CLI help for production
  scripts, use readable labels in figures and tables, record units and axis or
  orientation conventions, and avoid relying on color alone to communicate an
  important distinction.
- Add comments and docstrings where they explain scientific intent, data-axis or
  geometry conventions, provenance requirements, non-obvious invariants, or why
  an implementation choice is necessary. Keep straightforward mechanics
  self-explanatory through clear names; do not add line-by-line narration or
  comments that merely repeat the code.
- When changing structure, preserve compatibility where practical and document
  relocations. Remove or archive obsolete material only after checking manifests,
  hashes, references, and downstream consumers.
