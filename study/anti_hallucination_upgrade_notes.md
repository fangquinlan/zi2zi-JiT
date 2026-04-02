# Anti-Hallucination Upgrade Notes

This project originally follows a simple JiT-style recipe:

- predict clean glyph images directly with diffusion regression
- condition on font/style/content image cues
- keep the conditioning path lightweight

That design is strong for image quality and training simplicity, but it is not naturally strong at "symbol correctness". For Hanzi generation, a one-stroke error can change the character identity while still producing only a small pixel loss. The upgrades in this branch therefore try to add structure-aware supervision without discarding the original JiT backbone.

## What Does Not Conflict With JiT

- Unicode-aware shared character labels
- Explicit character embedding
- Multi-reference style aggregation
- Low-weight binary / edge / projection auxiliary losses

These changes keep the same denoising objective and the same core Transformer backbone. They mainly improve the conditioning signal and make the loss less blind to structural mistakes.

## What Partially Conflicts With JiT's Original Simplicity

- IDS conditioning
- Character-consistency auxiliary classification loss
- IDS token-presence loss

These additions do not conflict with "direct clean-image prediction" itself, but they do make the conditioning space more structured and less minimal. In practice this is acceptable for the current task because the project goal is not just image realism, but also character correctness.

## Why IDS Should Be Soft In Calligraphy

Historical calligraphy samples may differ from modern standard decomposition in subtle but legitimate ways. Because of that:

- IDS is useful as a semantic anchor
- IDS should not become an excessively hard constraint on every training sample
- `ids_lv2.txt` is a better default than lower-level files because it merges part of the must-unifiable variants

For this reason, the `calligraphy` upgrade profile keeps IDS conditioning on, but sets a relatively low IDS loss weight.

## Recommended Profiles

- `stable`
  - safest first upgrade for general finetuning
  - no IDS dependency
- `semantic`
  - stronger semantic supervision
  - still avoids IDS mismatch risk
- `ids`
  - strongest structure prior for standard Hanzi generation
- `calligraphy`
  - preferred for your current dataset
  - keeps IDS soft instead of forcing it too hard

## Maturity Rules For Future Changes

- Do not replace the original training path; keep all upgrades opt-in.
- Treat Unicode label consistency as the foundation, not as an optional afterthought.
- Prefer low auxiliary weights first; increase only after checking preview grids.
- For historical/calligraphic data, prefer soft structural priors over hard canonicalization.
- If a new semantic head is added, make sure it stays trainable under LoRA-only finetuning.
