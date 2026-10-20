# Pre-registration freezes

Each study in the paper (Table 4) was pre-registered: its rules, thresholds and analysis code were committed before the
first treatment run. The commits live in our **private development repository**; they are available to reviewers on
request. This file records, for each freeze and amendment, when it was committed, when GitHub first received it, and
when the first scored run was submitted, so the ordering can be checked.

Sources: commit times from git; *first on origin* from GitHub's repository activity API (push events; the earliest push
whose new commits include the freeze); *first treatment run* from Kaggle's kernel list (`lastRunTime`, which is the
submission time of a kernel's latest version, so an earlier version may have run before it). All times UTC.

| study | internal ID | prereg | freeze commit | committed | first on origin | branch | first treatment run | notes |
|---|---|---|---|---|---|---|---|---|
| S1 | v162 | `preregs/prereg_v162_parcae_truncation_training_rerun.md` | `89d8758` | 2026-10-01T18:12:44Z | 2026-10-03T09:30:38Z | feature/documentation | 2026-10-01T18:12:57Z (v162-tune-c/d) | Pushed after its runs (as the paper states). |
| S2 | v163 | `preregs/prereg_v163_parcae_damped_truncation.md` | `b9bf35c` | 2026-10-03T02:03:13Z | 2026-10-03T09:30:38Z | feature/documentation | 2026-10-03T02:03:22Z (v163-tune-c/d) | Pushed after its runs. Pilots v163-pilot-a/b (00:38Z) preceded the freeze. |
| S3 | v164 | `preregs/prereg_v164_parcae_prediction.md` | `1c5380c` | 2026-10-03T09:30:25Z | 2026-10-03T09:30:38Z | feature/documentation | 2026-10-03T09:30:29Z (v164-eval-a/c/d (09:30:29–35Z)) | Push came 3–9 s after the three evaluation kernels were submitted (stated in the paper, App. B). |
| S4 | v165 | `preregs/prereg_v165_undamped_rerun.md` | `6cec94e` | 2026-10-03T17:22:45Z | 2026-10-05T21:49:23Z | feature/documentation | 2026-10-03T17:22:53Z (v165-rerun-c) | Pushed after its runs. |
| S5 | v166 | `preregs/prereg_v166_slam_dunk.md` | `d0a0715` | 2026-10-04T00:46:03Z | 2026-10-05T21:49:23Z | feature/documentation | 2026-10-04T00:46:06Z (v166-eval-a–d) | Pushed after its runs. |
| S6 | v167 | `preregs/prereg_v167_retrofit_transfer.md` | `880be7a` | 2026-10-04T18:38:30Z | 2026-10-05T21:49:23Z | feature/documentation | 2026-10-04T18:38:38Z (v167-eval-c/d) | Pushed after its runs; the amendment was committed after the kernels were submitted but before any result. |
| S6 am. | v167 |  | `33fc028` | 2026-10-04T18:44:20Z | 2026-10-05T21:49:23Z | feature/documentation |  |  |
| S7 | v170 | `preregs/prereg_v170_sgd_vs_adam.md` | `29ecbb2` | 2026-10-05T04:44:43Z | 2026-10-05T21:49:23Z | feature/documentation | 2026-10-05T05:31:21Z (v170-eval-retro-c/d) | Pushed after its runs. |
| S7 am. | v170 |  | `1c67bdf` | 2026-10-05T04:53:46Z | 2026-10-05T21:49:23Z | feature/documentation |  |  |
| S8 | v174 | `preregs/prereg_v174_direction_intervention.md` | `095bd38` | 2026-10-11T02:53:37Z | 2026-10-11T02:53:48Z | feature/documentation | 2026-10-11T02:54:00Z (v174-scored-1/2) | Earlier pilot/gate kernels (v174-p5b, -gate5) preceded the freeze, as the prereg allows. |
| S9 | v195 | `preregs/prereg_v195_rho_modes.md` | `c1d8de0` | 2026-10-14T00:45:48Z | 2026-10-14T00:46:00Z | feature/documentation | 2026-10-14T00:46:13Z (v195-rho-modes) |  |
| S10 | v196 | `preregs/prereg_v196_band.md` | `24a61a8` | 2026-10-14T17:23:16Z | 2026-10-14T17:23:17Z | feature/documentation | 2026-10-14T17:23:29Z (v196-band) |  |
| S11 | v200 | `preregs/prereg_v200_ifm_transfer.md` | `4849a02` | 2026-10-15T01:23:49Z | 2026-10-15T01:23:53Z | feature/documentation | 2026-10-15T03:07:54Z (v200-*-study0) | LR pilots and the seed-spread pilot preceded the freeze; only the spread was read (paper, App. B). |
| S12 | v201 | `preregs/prereg_v201_batch_lost.md` | `38b3203` | 2026-10-15T06:45:51Z | 2026-10-15T06:46:03Z | feature/documentation | 2026-10-15T06:47:14Z (v201-parcae) | build_kaggle_v201.py was edited 2 min after the freeze to move kernels between Kaggle accounts (operational; frozen version in code/frozen_versions/). |
| S12 am. | v201 |  | `169d8cd` | 2026-10-15T07:49:32Z | 2026-10-15T07:49:33Z | feature/documentation |  |  |
| S13 | v203 | `preregs/prereg_v203_heldout_lost.md` | `97d945f` | 2026-10-15T21:22:50Z | 2026-10-15T21:22:52Z | feature/documentation | 2026-10-18T07:07:13Z (v203-parcae, v203-retro (version 2; see notes)) | Version 1 kernels died at parse (builder bug), fixed by amendment 1; amendment 2 replaced the Parcae fine-tune seed before any Parcae stage-300 data. Frozen versions in code/frozen_versions/. |
| S13 am.1 | v203 |  | `78716a1` | 2026-10-18T07:07:02Z | 2026-10-18T07:07:07Z | feature/documentation |  |  |
| S13 am.2 | v203 |  | `fc2e374` | 2026-10-18T08:30:19Z | 2026-10-18T08:30:22Z | feature/documentation |  |  |
| S14 | v204 | `preregs/prereg_v204_drift_prediction.md` | `e35d92f` | 2026-10-15T22:51:08Z | 2026-10-15T22:51:10Z | feature/documentation | 2026-10-18T06:36:37Z (v204-drift (latest version)) | Kaggle reports only the latest kernel version time. |
| S14 am. | v204 |  | `f767b13` | 2026-10-15T23:26:30Z | 2026-10-15T23:26:31Z | feature/documentation |  |  |
| S15 | v210 | `preregs/prereg_v210_followups.md` | `5cb861e` | 2026-10-19T15:59:59Z | 2026-10-19T16:00:04Z | feature/documentation | 2026-10-19T16:00:19Z (v210-parcae, -ouro, -ifm (16:00:19–31Z)) |  |
| S16 | v211 | `preregs/prereg_v211_k0_control.md` | `07446dd` | 2026-10-19T18:27:16Z | 2026-10-19T18:27:35Z | feature/documentation | 2026-10-19T18:28:47Z (v211-ifm; v211-parcae re-pushed unchanged 20:13:59Z) | The ifm kernel ran on a different Kaggle account than the prereg listed; the Parcae kernel sat queued ~1 h 40 min and was re-pushed unchanged. |

Amendment rows ("am.") are later commits to the same prereg, each made before the data it governs (see the prereg).

`tools/verify_hashes.py` checks the code hashes recorded in each prereg against `code/` (and `code/frozen_versions/`
for versions later changed by an amendment or an operational edit).
