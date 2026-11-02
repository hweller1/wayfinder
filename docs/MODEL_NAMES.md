# Model names

| Display name | What it is | Internal ids (unchanged in code, rows and file names) |
|---|---|---|
| wayfinder-glimmer | Muse Glimmer 30B post-trained with LoRA SFT as a code-search subagent (called "v3" or "SFT v3" in older files) | served model name `wayfinder-v3` on Modal (`wayfinder-glimmer` on the GCP VM); Modal app `wayfinder-v3-serve`; Modal Volume path `/data/sft/v3_merged`; artifact folders and result-row keys `v3`; Scout gateway profile `v3` |

In the Scout plugin, `/scout:model glimmer` is an alias for the `v3` profile. Only the display name changed; ids were left alone because renaming them would break existing result rows and scorers. Other things named "v3" are not the model: task ids with the `v3__` prefix, RL pool v3, `hard_v3`, SFT data such as `teacher_v3`, and arms such as `super_v3`. SFT iterations v1, v2a, v2b and v4 keep their names.

The result documents in `docs/results/` are historical records written during the event and use these ids. A second model, wayfinder-qwen, is in progress and is not part of this release.
