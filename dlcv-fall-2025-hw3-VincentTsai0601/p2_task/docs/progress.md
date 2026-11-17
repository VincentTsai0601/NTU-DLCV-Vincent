# SDD ledger — plan: docs/superpowers/plans/2026-10-25-captioning.md

Pre-flight: Task 1 produces model/checkpoint contracts consumed by Task 2; metadata must fully specify inference architecture. Task 3 documents those same CLIs.

Ruling: This is a standalone uploaded-file project without a git repository; use an isolated project directory and retain the ledger in the delivered source archive, rather than creating an unrelated git worktree.

Ruling: The user authorized continuing implementation after the proposed ViT/projector/LoRA approach; execute it directly without asking for another confirmation.

Task 1: complete — pretrained-output, loss, gradients, padding, KV-cache and adapter tests pass.
Task 2: complete — real tiny decoder training, adapter reload, image caption inference and JSON output pass; token-weighted accumulation matches single-batch optimization.
Final review: fresh read-only reviewer found a standard local vision checkpoint classifier-head mismatch. Fixed through timm pretrained file overlay; regression test observed failing before the fix.
Final: fixed missing mixed-EOS coverage before final review response — deterministic real-projection test passes; CPU bf16 autocast also passes.
Final: fixed incompatible latest Hub dependency — observed timm import failure with Hub 2.3.0; pinned 0.25.2 because it exports the required import.
Final: minor (deferred): hw3_2.sh uses the timm cache and does not expose --vision_path; offline users must pre-populate that cache or adapt the wrapper.
Final: Ruling: no full-size GPU quality, memory, timing, or decoder-checkpoint claims — original weights/data and target GPU were not provided; validate these on the user machine.

Final verification: 19/19 CPU tests pass; both CLI --help commands and compileall pass; bash syntax passes. Actual pretrained CLIP ViT downloaded and loaded, with feature shape (1,197,768). Supplied modules verified byte-identical.
Task 3: complete — README, pinned requirements, tests and grading wrapper included in the delivery archive.
