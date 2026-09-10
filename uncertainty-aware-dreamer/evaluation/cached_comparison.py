"""Load Gaussian ensembles, validate cached priors, and rescore fixed inputs.

load_ensemble strictly restores saved MLP weights without a simulator.
cache_groups validates the two original prior cache formats and their pairing.
score_inputs shares one ensemble forward across metrics and isolates JS RNG.
physical_discrepancy reproduces the original transformed-state position error.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from uncertainty_aware_dreamer.envs.dmc.suite_env import SuiteBaseEnv
from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.ensemble import EnsembleModel
from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.distance_measure import (
    GeometricJensenShannonDivergence, JensenShannonDivergence, TransitionMeanVariance)


def sha256(path):
    """Identify source files by streamed SHA-256 without changing them."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ensemble(run_dir, feature_size, action_size, device):
    """Restore exactly the saved architecture/target; leave config.log_dir intact."""
    run_dir = Path(run_dir)
    config = OmegaConf.load(run_dir / "config.yaml")
    if config.algorithm.name != "li-urssm":
        raise ValueError("Only Gaussian li-urssm caches are supported; categorical needs separate validation")
    transition = config.algorithm.world_model.model.transition
    if transition.type != "r_rssm" or feature_size != transition.lsd + transition.rec_state_dim:
        raise ValueError("Cache feature dimensions do not match the saved Gaussian transition")
    checkpoint = torch.load(run_dir / "networks.pth", map_location="cpu", weights_only=True)
    weights = checkpoint["ensemble"]
    heads = [v for k, v in weights.items() if k.startswith("_models.0.")
             and "._mean_net" in k and k.endswith(".weight") and v.ndim == 2]
    if len(heads) != 1:
        raise ValueError("Cannot identify ensemble mean head")
    # Initialization consumes RNG but must not affect any caller's random stream.
    with torch.random.fork_rng(devices=[]):
        ensemble = EnsembleModel(feature_size, transition.lsd, transition.rec_state_dim,
                                 heads[0].shape[0], action_size, False, config.algorithm.ensemble)
    ensemble.load_state_dict(weights, strict=True)
    ensemble.to(device).eval()
    return ensemble, config


def cache_groups(cache):
    """Yield free-running prior groups only; never shift cached actions again.

    get_priors(include_first=True) returns z_0..z_(T-1), a_0..a_(T-1).
    The last action's successor is absent from these length-T caches.
    """
    if not isinstance(cache, dict):
        raise ValueError("Expected an infos dictionary")
    if "prior" in cache:
        groups = [(f"prior_{name}", cache["prior"][name]) for name in ("closed", "open")
                  if name in cache["prior"]]
    elif {"states", "act", "dec_phys_states", "gt_phys_states"} <= cache.keys():
        groups = [("prior", cache)]
    else:
        raise ValueError("Unsupported cache; expected prior_infos or random_rollouts prior branches")
    if not groups:
        raise ValueError("No prior trajectories found")
    for name, group in groups:
        states, actions = group["states"], group["act"]
        sample, recurrent = states["sample"], states["gru_cell_state"]
        for value in (sample, recurrent, actions):
            if not isinstance(value, torch.Tensor) or value.ndim != 3 or min(value.shape) < 1:
                raise ValueError("Expected nonempty [trajectory, time, feature] tensors")
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite cached conditioning input")
        if sample.shape[:2] != actions.shape[:2] or recurrent.shape[:2] != actions.shape[:2]:
            raise ValueError("State/action lengths differ; refusing to infer or repair alignment")
        # RRSSMTM.get_features uses sample, not the deterministic mean.
        yield name, torch.cat((sample, recurrent), -1).cpu(), actions.cpu(), group


@torch.no_grad()
def score_inputs(ensemble, features, actions, metrics, batch_size=256,
                 js_num_samples=128, js_chunk_size=8, js_seed=0, js_repeats=5):
    """Return [N,T] scores, [R,N,T] JS repeats, and shared [K,N,T,D] predictions."""
    if min(batch_size, js_repeats, js_num_samples, js_chunk_size) < 1:
        raise ValueError("Batch, repeat, sample, and chunk counts must be positive")
    if not metrics or set(metrics) - {"gjs", "js", "transition_var"}:
        raise ValueError("Unsupported metric selection")
    parameter = next(ensemble.parameters())
    device = parameter.device
    shape = features.shape[:2]
    flat_features, flat_actions = features.flatten(0, 1), actions.flatten(0, 1)
    measures = {"gjs": GeometricJensenShannonDivergence(),
                "transition_var": TransitionMeanVariance(),
                "js": JensenShannonDivergence(js_num_samples, js_chunk_size)}
    scores = {name: [] for name in metrics}
    means, stds, repeats = [], [], []
    devices = [device.index] if device.type == "cuda" else []
    for batch_index, start in enumerate(range(0, len(flat_features), batch_size)):
        stop = start + batch_size
        predictions = ensemble(flat_features[start:stop].to(parameter), flat_actions[start:stop].to(parameter))
        mean, std = predictions["mean"], predictions["std"]
        means.append(mean.cpu())
        stds.append(std.cpu())
        variance = std.square()
        for name in metrics:
            if name == "js":
                estimates = []
                for repeat in range(js_repeats):
                    # Local generators provide states without seeding any other device.
                    seed = (js_seed + batch_index * js_repeats + repeat) % (2**63 - 1)
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.random.fork_rng(devices=devices):
                        if device.type == "cuda":
                            torch.cuda.set_rng_state(generator.get_state(), device)
                        else:
                            torch.random.set_rng_state(generator.get_state())
                        estimates.append(measures[name].compute_measure(mean, variance).squeeze(-1).cpu())
                values = torch.stack(estimates)
                repeats.append(values)
                scores[name].append(values.mean(0))
            else:
                scores[name].append(measures[name].compute_measure(mean, variance).squeeze(-1).cpu())
    scores = {name: torch.cat(parts).reshape(shape).numpy() for name, parts in scores.items()}
    repeat_values = torch.cat(repeats, 1).reshape(js_repeats, *shape).numpy() if repeats else None
    mean = torch.cat(means, 1).reshape(len(ensemble._models), *shape, -1).numpy()
    std = torch.cat(stds, 1).reshape(mean.shape).numpy()
    return scores, repeat_values, mean, std


def physical_discrepancy(group, config):
    """Match broad_analysis.compute_phys_diff, including its masked dimension denominator.

    Reuses SuiteBaseEnv metadata without instantiating it. Inverts the exact
    sin/cos encoding in StateBasedDMCMBRLEnv.untransform_phys_state, masks raw
    translation coordinates, keeps the first half (positions), wraps angles,
    and averages absolute differences including the zeroed position dimensions.
    """
    if "gt_phys_states" not in group:
        return None, "UQ only: random_rollouts has no full action-matched physical ground truth"
    env_name = OmegaConf.select(config, "environment.env.env")
    domains = [name for name in SuiteBaseEnv.DMC_ENV_CLASSES if env_name and env_name.startswith(name + "_")]
    if len(domains) != 1:
        raise ValueError("Unknown physical environment metadata; use --uq-only explicitly")
    metadata = SuiteBaseEnv.DMC_ENV_CLASSES[domains[0]]
    angles = metadata.IS_ANGLE
    positions = len(metadata.RANGES)
    mask = torch.as_tensor(metadata.get_translation_inv_mask(False))
    expected = positions * 2 + sum(angles)
    if len(mask) != positions * 2 or len(angles) != positions:
        raise ValueError("Inconsistent original physical metadata; use --uq-only")
    transformed = []
    for key in ("dec_phys_states", "gt_phys_states"):
        state = group[key].detach().cpu()
        if state.ndim != 3 or state.shape[-1] != expected or not torch.isfinite(state).all():
            raise ValueError(f"Invalid transformed physical state: {key}; expected last dimension {expected}")
        raw = []
        for dim in range(positions * 2):
            offset = dim + sum(angles[:dim])
            raw.append(torch.atan2(state[..., offset], state[..., offset + 1])
                       if dim < positions and angles[dim] else state[..., offset])
        transformed.append((torch.stack(raw, -1) * mask)[..., :positions])
    if transformed[0].shape != transformed[1].shape or transformed[0].shape[:2] != group["act"].shape[:2]:
        raise ValueError("Physical ground truth is not aligned to cached states")
    # NumPy modulo matches the original implementation, including angle boundaries.
    delta = (transformed[0] - transformed[1]).numpy()
    for dim, is_angle in enumerate(angles):
        if is_angle:
            delta[..., dim] = np.mod(delta[..., dim] + np.pi, 2 * np.pi) - np.pi
    return np.abs(delta).mean(-1), "Original mean absolute masked position/angle discrepancy at state t"
