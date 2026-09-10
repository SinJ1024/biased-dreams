# Configurable ensemble uncertainty and fixed-input visualization

This guide describes training overrides, metric definitions, checkpoint
compatibility, fixed-input plotting, and a separate CVaR risk analysis.
For immutable cross-metric scoring of existing prior caches, see
[CACHED_COMPARISON.md](CACHED_COMPARISON.md).
The implementation is local to uncertainty-aware-dreamer; infoprop remains
independent and unchanged.

## Training

Run from uncertainty-aware-dreamer in its training environment:

```bash
python main.py algorithm=li-urssm environment=cheetah_run algorithm.ensemble.distance_measure=gjs
python main.py algorithm=li-urssm environment=cheetah_run algorithm.ensemble.distance_measure=js algorithm.ensemble.js_num_samples=128 algorithm.ensemble.js_sample_chunk_size=8
python main.py algorithm=li-urssm environment=cheetah_run algorithm.ensemble.distance_measure=transition_var
```

The same overrides work with algorithm=li-cat_urssm. The li-rssm and
li-cat_rssm baselines do not use this ensemble. Legacy jrd remains available.
The default is still gjs, with its original arithmetic and random-number usage.
No model layers, training losses, optimizer settings, or weight keys changed.
Old configurations without JS settings use 32 samples per member and chunks of 8
when selecting JS. The measures contain no parameters or persistent buffers,
so ensemble state_dict checkpoints load strictly across choices. Preserve the
saved architecture configuration when loading weights. This does not add exact
training resumption: the original networks.pth does not save optimizer/RNG state.

## Definitions and cost

Let K be the number of members, D the output dimension, and B the number of
conditioning inputs. Each member predicts a diagonal Gaussian with mean mu_k
and variance v_k. Inputs have shape [K, ..., D]; outputs have shape [..., 1].

| Choice | Definition | Approximate arithmetic cost |
| --- | --- | --- |
| gjs | Original average of pairwise geometric JS divergences | O(B K^2 D) |
| js | (1/K) sum_k E_p_k[log p_k(Y) - log ((1/K) sum_j p_j(Y))] | O(B S K^2 D) |
| transition_var | (1/D) sum_d (1/K) sum_k (mu_kd - mean_j mu_jd)^2 | O(B K D) |

JS is the generalized equal-weight Jensen-Shannon divergence in natural-log
units, not an average of pairwise arithmetic JS. S is js_num_samples **per
member per input**, so each input uses K*S samples. The mixture is evaluated
with logaddexp; diagonal Gaussian evaluation needs no dense inverse or
determinant. Sample chunks limit temporary storage to approximately O(C B D),
with C=js_sample_chunk_size, in the no-grad training scoring path. Python/TorchScript
loop overhead also matters; benchmark at the actual device and batch size.

The true JS lies between 0 and log(K). The Monte Carlo estimator is stochastic
and can be slightly negative; values are deliberately not clamped to avoid
adding truncation bias. It uses the existing PyTorch RNG on the tensor's device.
Fix the seed, sample count, chunk size, software, device, and call order for
repeatability. Switching to JS consumes random numbers and changes training
trajectories; it is an experimental configuration, not bitwise reproduction of
a GJS run. More samples reduce estimation noise but increase cost.

transition_var uses population variance (correction=0) and averages output
coordinates. It excludes within-member predictive variance. Even when the
target is deter or embed, it retains this precise definition; it is not the
variance of a sampled next physical state. It is scale dependent. GJS, JS, and
transition_var have different ranges/units, so any uncertainty reward or penalty
coefficient needs separate calibration for a fair experiment.

## Fixed state-action plots

The plotting API accepts the ensemble's actual feature vector and action.
It requires exactly five members and one input, preserves model train/eval
modes, and produces analytic two-coordinate Gaussian **marginals**, not slices
through higher-dimensional density or pooled predictions. Remaining output
coordinates are integrated out. The optional third axis is probability density.
Projection can hide disagreement in unselected coordinates; try multiple pairs.
Coordinate values are in the ensemble target space, not decoded physical units.

In a loaded experiment, select exactly one aligned feature/action pair:

```python
import numpy as np
from evaluation.ensemble_distribution import plot_fixed_input

# Match LatentImaginationTrainer.compute_uncertainty(align=True).
features = experiment._model.get_features(state=imagined_states)
batch_index, time_index = 0, 0
state = features[batch_index, time_index].detach()
action = actions[batch_index, time_index + 1].detach()
plot_fixed_input(experiment._ensemble, state, action, "plots/fixed_input",
                 dimensions=(0, 1), surface=True)
np.savez("fixed_input.npz", state=state.cpu().numpy(), action=action.cpu().numpy())
```

Here imagined_states/actions are tensors from the same imagination rollout,
not arbitrary unrelated observations or an environment state substituted for
RSSM features. For already aligned state-action data, use matching indices
without the +1 shift. Detach/copy the exact input you want to inspect.

You can later load only the saved MLPs, without constructing a simulator:

```bash
python plot_ensemble.py --run-dir path/to/run --input fixed_input.npz --dimensions 0 1 --surface --output-prefix plots/fixed_input
```

The run directory must contain config.yaml and networks.pth. The NPZ must contain
one-dimensional state and action arrays. The CLI reconstructs the output size
from the checkpoint's Gaussian mean head and loads ensemble weights strictly.
It writes *_contours.png, optional *_surfaces.png, and *_predictions.npz with
the exact conditioning input, all five means/stds, selected dimensions, and
the density grid. Contours enclose 50%, 80%, and 95% of each marginal's mass;
all five Gaussians are overlaid on a single set of axes in each figure, with
consistent member colors and legends. The 3D surfaces use transparency and
95% ground-plane contours. Surface values below 1% of each member's peak are
hidden to reduce occlusion; the archived density arrays remain complete.
Increase the API grid_size for very narrow distributions on a broad common grid.

## CVaR as a separate risk measure

First define a scalar loss L, such as negative discounted return, constraint
violation cost, or a task-relevant prediction error. For confidence alpha in
(0, 1), upper-tail loss CVaR is

```text
CVaR_alpha(L) = min_t {t + E[max(L - t, 0)] / (1 - alpha)}.
```

For continuous distributions it is the mean loss in the worst 1-alpha fraction.
The optimization definition also handles atoms correctly; simply averaging
all samples >= a quantile can mishandle ties. For rewards, apply this definition
to loss=-return and reverse the sign when reporting worst-tail return.
See [Rockafellar and Uryasev, Conditional value-at-risk for general loss distributions](https://www.sciencedirect.com/science/article/pii/S0378426602002716).

CVaR can distinguish losses with equal means and variances but different tails.
For example, a symmetric loss taking +/-1 with equal probabilities and a loss
taking -10/0/+10 with probabilities 0.005/0.99/0.005 both have mean zero and
variance one. Their 99% upper-tail CVaRs are 1 and 5, respectively. This example
uses the atom-aware definition above. A fixed CVaR level still does not fully
characterize a distribution, nor is it a general test for heavy tails.

For this repository, CVaR is valuable as an additional task-risk evaluation,
not a synonymous replacement for epistemic disagreement:

- Mixture loss CVaR combines within-member randomness and between-member
  variation. It can be large even if all members agree exactly.
- CVaR across five member expected losses excludes much within-member noise,
  but at 95% or 99% confidence it is just the worst member under equal weights.
  Five members cannot resolve a rare tail; bootstrap intervals and additional
  trajectory samples are needed for meaningful empirical tail evaluation.
- Each current member is Gaussian. A linear loss of one Gaussian has tail risk
  determined by its mean and standard deviation; CVaR cannot recover unmodeled
  heavy tails. Finite Gaussian mixtures may have greater kurtosis or separated
  modes but are still asymptotically light-tailed. Consider calibrated residual
  distributions, Student-t/mixture heads, or distributional returns in a separate
  experiment if tail misspecification is the central issue.
- Sorting N already available scalar losses costs O(N log N), with no density
  matrix calculation. Generating enough credible tail trajectories may dominate
  that saving. Only about N*(1-alpha) samples lie in the requested tail, and
  correlated rollouts reduce effective sample size further.

Suggested evaluation: retain one epistemic score, separately log CVaR of task
loss at several levels, and compare against held-out realized failures/returns.
Condition comparisons on the same initial state-action and policy, distinguish
member sampling from transition randomness, and report seed variability and
wall-clock cost. No CVaR training objective or new distribution head is introduced
by this change.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover JS against quadrature, identical/separated members, tensor shapes,
sample-count validation, reproducible sampling, mean-variance semantics,
legacy GJS arithmetic, strict weight loading across metrics, finite training
gradients, mode restoration, mixed-input rejection, and normalized plot grids.

Local validation passed eight tests on Windows with Python 3.13 and PyTorch
2.13 CPU, including the saved-checkpoint plotting CLI. Legacy GJS and JRD class
implementations were also checked against Git HEAD and are AST-identical.
Plot layouts were visually inspected using randomly initialized MLPs. No trained
checkpoint was present locally; full DMC training, CUDA behavior, and the pinned
Python 3.12/PyTorch 2.6 environment have not been executed in this validation.
