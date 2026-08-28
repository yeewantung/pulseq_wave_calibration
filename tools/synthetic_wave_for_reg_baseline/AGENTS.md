# Local development rules

These rules apply to work under `tools/synthetic_wave_for_reg_baseline/`.

## Code and output organization

- Always use BART's GPU option (`-g`) for every reconstruction. Do not silently
  fall back to CPU. If a required BART command does not support `-g`, stop and
  document that command-specific exception before proceeding.
- Never put workflow phase or step numbers in code filenames, test filenames,
  generated-result filenames, or output-directory names. Name artifacts for
  their stable scientific purpose or operation instead; phase numbering changes
  over time and makes code, links, manifests, and result trees hard to maintain.
- Before adding code or generated outputs, choose a clear location within the
  existing structure. Keep reusable implementation in `scripts/`, focused tests
  in `tests/`, documentation in `docs/`, and task-specific requirements in
  `requirements/`. Do not place large generated artifacts in the source tree.
- Put generated experiment artifacts in a separate, descriptively named dataset
  output tree. Never overwrite accepted or historical results. Keep each run
  discoverable through a manifest or index that records its purpose, status,
  inputs, configuration, provenance, and canonical outputs.
- Never expose an actual machine-specific data, user-home, environment, scan,
  or generated-output path to public GitHub. This prohibition applies to every
  tracked file and staged change, including source, documentation, examples,
  tests, fixtures, manifest snapshots, and logs. Put actual paths only in
  matching `.local.sh` or `.local.json` files that are explicitly ignored by
  Git; verify them with `git check-ignore` before use. Keep tracked runners
  path-agnostic, use placeholders in copyable `.example.*` files and environment
  variables in public documentation, and audit tracked content plus the staged
  diff for private-path signatures before every commit or push.
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
- Keep code paths concise and make the scientific operation obvious at first
  reading. Before writing an adapter or runner, find and reuse the existing
  function that owns the operation; do not reimplement algorithms, helpers,
  I/O, validation, or orchestration that already exist. When a workflow only
  needs one existing function call, keep its wrapper thin and put the exact
  imported callable, implementation file, and backend in a plain sentence at
  the top and beside the call. Never leave it ambiguous whether a result uses
  the local/Torch implementation, BART, or another backend. Add brief comments
  only where they are necessary to explain non-obvious data adaptation or
  scientific choices, and keep provenance/export plumbing visibly separate
  from the core reconstruction call.
- When changing structure, preserve compatibility where practical and document
  relocations. Remove or archive obsolete material only after checking manifests,
  hashes, references, and downstream consumers.
