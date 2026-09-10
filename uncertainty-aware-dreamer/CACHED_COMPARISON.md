# Same-checkpoint comparison on cached latent trajectories

This guide explains compare_uncertainty.py, cache alignment, physical error,
output statistics, reproducibility, and the unsubmitted Slurm template.

## Scope and loading

The CLI supports Gaussian li-urssm. It reads the original config.yaml and the
ensemble state_dict in networks.pth, preserving architecture, target, and weights
with strict=True. It does not instantiate a simulator or RSSM, decode new states,
sample new latent trajectories, select starts, train, or edit config.log_dir.
It uses cached decoded physical states and cached action-matched ground truth.
RSSM features are exactly concat(sample, gru_cell_state), as in
RRSSMTM.get_features; using mean instead of sample would change the experiment.
The saved transition dimensions are checked against individual cached components.

The provided list of cluster files did not confirm config.yaml. That original
resolved training configuration is also required alongside networks.pth. If it
is missing, recover it from the original run; do not invent architecture settings.
Categorical li-cat_urssm is explicitly rejected pending separate validation.

The existing saved caches have no checkpoint identifier. The comparison records
hashes of the supplied checkpoint and caches but cannot prove they were generated
together. The operator must confirm this provenance. Scoring a cache with a
different model would not meet the intended experimental design.

## Exact time indexing from the generating code

- evaluation/utils/data_collector.py: imagine_rollout and
  prior_rollout_from_actions prepend z_0 and remove the final successor when
  include_first=True. The returned cache is z_0,...,z_(T-1) with a_0,...,a_(T-1).
- evaluation/common/decoding.py: get_uncertainties calls compute_uncertainty
  with align=False. Thus use cached state t and cached action t without shifting.
- evaluation/common/env_simulation.py: get_env_infos_from_actions records the
  initial physical state and executes the first T-1 actions. Its physical state
  at t is reached after actions a_0,...,a_(t-1).
- evaluation/strategy/broad_analysis.py: analyze_physical_discrepancy saves
  states, act, dec_phys_states, gt_phys_states, and the original unc in prior_infos.
  Decoder output at t and cached physical ground truth at t correspond directly.
  Posterior caches follow a different observation-update path and are rejected.

The four-row plot uses the same index t for state discrepancy(t) and UQ(z_t,a_t).
These are aligned time trends, not an assertion that UQ(t) predicts error(t).
The separate association statistic pairs UQ(t) with discrepancy(t+1), omitting
the last action because its successor is absent. Even that successor discrepancy
is an accumulated free-running prediction error, not a teacher-forced local
one-step transition error. Each trajectory receives its own descriptive Pearson
correlation, with null for constant series or fewer than two pairs. There are no
IID p-values for autocorrelated timesteps. Endpoint error/UQ changes are also
saved as descriptive evidence, without a hard-coded success/failure verdict.

random_rollouts.pt has post and prior.open/prior.closed branches. Only the two
free-running prior branches are scored, separately. They contain dec_phys and
start_phys but no full physical ground truth. They therefore produce clearly
marked UQ-only output; start_phys is never broadcast as fake ground truth.
Filename checks additionally reject the original *_post_infos.pt files. Renaming
a posterior cache to a prior filename cannot establish correct provenance.

## Physical discrepancy

The implementation follows broad_analysis.compute_phys_diff and
StateBasedDMCMBRLEnv.untransform_phys_state and reuses metadata from the original
SuiteBaseEnv.DMC_ENV_CLASSES without constructing an environment:

1. Invert each encoded (sin(theta), cos(theta)) pair with atan2(sin, cos).
2. Apply get_translation_inv_mask(transformed=False) to raw physical coordinates.
3. Select the first half of the raw vector, i.e. position coordinates only.
4. For angles use abs((prediction - truth + pi) mod (2*pi) - pi); otherwise
   use absolute coordinate differences.
5. Average over **all** position dimensions, retaining zeroed translation
   dimensions in the denominator, exactly as the original metric does.

This is a mean absolute position/angle discrepancy, not generic tensor MSE.
The physical test compares directly with the original source functions compiled
in isolation to avoid importing simulator dependencies. Unknown environments,
inconsistent metadata, nonfinite physical values, or shape mismatches cause an
explicit failure. --uq-only intentionally disables physical error and labels it
as unavailable. No alternative error is silently substituted.

## Metrics and random numbers

For K members with diagonal Gaussian means mu and variances v:

- transition_var = mean_d mean_k (mu_kd - mean_j mu_jd)^2 (population variance).
- JS = mean_k E_p_k[log p_k(y) - log(mean_j p_j(y))], in nats. The CLI invokes
  the existing JensenShannonDivergence; js-num-samples is per member per input.
- GJS is the original average over unordered pairs. For each pair, let
  v_A = 1/(0.5/v_i + 0.5/v_j), mu_A = v_A*(0.5*mu_i/v_i + 0.5*mu_j/v_j).
  Its contribution is 0.5*sum_d(0.5*mu_i^2/v_i + 0.5*mu_j^2/v_j
  - mu_A^2/v_A + 0.5*log(v_i) + 0.5*log(v_j) - log(v_A)).

Each flattened batch runs the ensemble once. All metrics and JS repeats reuse
its predictions. The start/middle/end plots also reuse these predictions via
plot_fixed_input(precomputed_predictions=...), with no additional forward.
Inputs are restored to [trajectory, time] score arrays. Member predictions are
held on CPU for plotting; memory scales with K*N*T*D, so very large caches may
require splitting into complete trajectory groups in a separate preparation step.

JS uses a local torch.Generator to supply RNG state inside fork_rng. Only the CPU
or selected CUDA device stream is temporarily replaced, and it is restored on
exit. Local seed = (js_seed + batch_index*js_repeats + repeat) mod (2^63-1),
reset for each cache branch. This preserves RNG state outside rescoring and
records batch size, sample count, chunk size, repeat count, seed, and versions.
Changing batch size, chunk size, software, or device is not promised to preserve
bitwise results. Negative finite JS Monte Carlo estimates are not clamped.

## Outputs and interpretation

Use a new output directory outside the source run directory. A nonempty output
directory is refused, including a partial failed run. Each file/branch gets its
own subdirectory; different cache populations are not silently pooled.

- scores.csv: each trajectory and state index, physical discrepancy at t and
  t+1 when available, all requested UQ values, individual JS repeats and MC SD.
- scores.npz: [N,T] scores/errors and [R,N,T] JS repetitions.
- aggregate.json: per-time trajectory mean and population SD; separate MC SD
  of the trajectory-mean JS estimator across repeat indices.
- associations.json: descriptive per-trajectory successor-error correlations
  and endpoint changes. Undefined correlations are null, not zero.
- time_comparison.png: four rows with independent y axes for physical discrepancy,
  GJS, JS, and transition_var; unrequested rows are marked. Solid curves show
  trajectory means, shading shows +/-1 population SD across trajectories, and
  orange JS dashed lines show +/-1 MC SD of the trajectory mean. None is a
  confidence interval across independent training seeds. JS with one repeat has
  zero empirical MC SD and supplies no useful assessment of sampling variability.
- trajectory_* plots/NPZ: five Gaussian marginal overlays at t=0, floor(T/2),
  T-1 of the same selected trajectory. Coordinates and density limits are shared
  across these times within each group; two dimensions can hide disagreement in
  unselected coordinates. Use --plot-trajectory to inspect a different fixed row.
- manifest.json: status (including failure), checkpoint/cache/config SHA-256,
  source paths, git commit, analysis code hashes, settings, formulas and caveats.
  source_config.yaml is a byte-for-byte copy, and working_tree.patch records
  tracked changes relative to HEAD. Sources are rehashed after successful work.

The ID/OOD starts were selected with the original GJSD procedure. Report this as
**cross-metric comparison on trajectories selected by original GJSD**, not an
independent OOD label or unbiased benchmark. A score failing to increase as
physical error increases indicates a limitation on these cached trajectories;
it does not prove failure universally. Conversely, increasing scores do not
alone establish useful calibration or predictive discrimination.

Rescoring cannot change the already generated latent dynamics. Declining scores
alone are not proof of an attractor. No latent PCA or contraction claims are
added. Controlled contraction requires verified distinct starts, common action
sequences, a defined distance/reference, and an appropriate noise coupling;
the legacy caches do not record sufficient provenance to assert these controls.

## DelftBlue command (run manually in the prepared environment)

Dependencies: the existing Gaussian training environment, or Python >=3.10 with
PyTorch, NumPy, Matplotlib, and OmegaConf. The offline CLI needs no dm_control,
MuJoCo, simulator, scikit-learn, or new model training. Run from the repository's
uncertainty-aware-dreamer directory. Use CPU first or cuda in an allocated GPU
job; do not execute substantial analysis on a login node.

```bash
RUN=/scratch/jingyuansun/biased-dreams/uncertainty-aware-dreamer/out/paper-core-pd/li-urssm/cheetah_run/0
OUTPUT=/scratch/jingyuansun/biased-dreams/uncertainty-aware-dreamer/out/cached-comparison/li-urssm/cheetah_run/0-$(date -u +%Y%m%dT%H%M%SZ)
python compare_uncertainty.py \
  --run-dir "$RUN" \
  --infos "$RUN/id_prior_infos.pt" "$RUN/ood_prior_infos.pt" "$RUN/random_rollouts.pt" \
  --metrics gjs js transition_var \
  --output-dir "$OUTPUT" --device cpu --batch-size 256 \
  --js-num-samples 128 --js-chunk-size 8 --js-seed 2026 --js-repeats 5 \
  --dimensions 0 1 --plot-trajectory 0 --surface
```

scripts/compare_uncertainty.slurm is an unsubmitted template. No existing Slurm
script was found in this checkout. Account, partition, wall time, memory, CPU/GPU
resources, and module/conda activation are intentionally left to the operator.
Provide site-approved resource options to sbatch and set REPO_DIR and OUTPUT_DIR
before submission. The template defaults to CPU and requests no GPU model.

## Local verification

```bash
python -m unittest discover -s tests -v
```

Synthetic tests cover strict weight loading, unchanged config.log_dir, Gaussian
feature ordering and state/action pairing, shape recovery, one forward per batch,
CPU JS RNG restoration, deterministic repeatability, GJS/variance batch tolerance,
original physical error equivalence, unknown metadata rejection, prior/random
cache CLI output, shared plot grids, refusal to overwrite, and source-file hashes.
These tests do not certify that the real remote caches match the supplied remote
checkpoint. No DelftBlue job or real checkpoint experiment has been run locally.
