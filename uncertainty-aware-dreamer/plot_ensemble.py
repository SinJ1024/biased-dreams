"""CLI for fixed-input ensemble contours and optional probability density surfaces.

main reconstructs only the MLP ensemble from saved configuration and weight
shapes, loads its checkpoint strictly, and plots one supplied latent state-action.
No simulator, policy rollout, or changes to the training checkpoint are required.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.ensemble_distribution import plot_fixed_input
from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.ensemble import EnsembleModel


def main():
    """Load a run's config/networks and a singleton state/action NPZ, then plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with state and action vectors")
    parser.add_argument("--output-prefix", default="ensemble_distribution")
    parser.add_argument("--dimensions", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--surface", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(args.run_dir / "config.yaml")
    checkpoint = torch.load(args.run_dir / "networks.pth", map_location="cpu", weights_only=True)
    weights = checkpoint["ensemble"]
    ensemble_config = config.algorithm.ensemble
    with np.load(args.input, allow_pickle=False) as data:
        state, action = data["state"].copy(), data["action"].copy()
    if state.ndim != 1 or action.ndim != 1:
        raise ValueError("Input archive must contain one-dimensional state and action vectors")
    # The Gaussian mean head determines output size for every target type.
    mean_weights = [value for key, value in weights.items()
                    if key.startswith("_models.0.") and "._mean_net" in key
                    and key.endswith(".weight") and value.ndim == 2]
    if len(mean_weights) != 1:
        raise ValueError("Cannot identify the Gaussian mean head in this checkpoint")
    output_size = mean_weights[0].shape[0]
    ensemble = EnsembleModel(feature_size=state.size, action_size=action.size,
                             stoch_size=output_size, deter_size=output_size,
                             embed_size=output_size, dyn_discrete=False, config=ensemble_config)
    ensemble.load_state_dict(weights, strict=True)
    for path in plot_fixed_input(ensemble, state, action, args.output_prefix,
                                 dimensions=tuple(args.dimensions), surface=args.surface):
        print(path)


if __name__ == "__main__":
    main()
