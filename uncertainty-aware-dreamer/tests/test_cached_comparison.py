"""Synthetic-cache integration tests for immutable cross-metric comparison.

Tests check state/action alignment, strict checkpoint loading, single forwards,
RNG isolation, batching invariance, original physical conventions, and CLI plots.
Fixtures are synthetic and do not validate real DelftBlue checkpoint provenance.
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from compare_uncertainty import main
from evaluation.cached_comparison import cache_groups, load_ensemble, physical_discrepancy, score_inputs, sha256
from test_ensemble_uncertainty import make_ensemble
from uncertainty_aware_dreamer.envs.dmc.suite_env import SuiteBaseEnv


def fixture(directory):
    """Write an original-format prior cache and a tiny real ensemble checkpoint."""
    run = directory / "run"
    run.mkdir()
    ensemble_config = dict(target="deter", num_ensemble=5, hidden_size=8, activation="ELU",
        normalization="LayerNorm", num_layers=2, init_std=1.0, min_std=0.1, max_std=5.0,
        sigmoid_activation=False, output_normalization="none", distance_measure="gjs")
    config = OmegaConf.create({"algorithm": {"name": "li-urssm", "ensemble": ensemble_config,
        "world_model": {"model": {"transition": {"type": "r_rssm", "lsd": 1, "rec_state_dim": 2}}}},
        "environment": {"env": {"env": "cheetah_run"}}, "log_dir": "/original/remote/run"})
    OmegaConf.save(config, run / "config.yaml")
    torch.manual_seed(9)
    ensemble = make_ensemble()
    torch.save({"ensemble": ensemble.state_dict()}, run / "networks.pth")
    # Distinct actions make any accidental time shift observable.
    cache = {"states": {"sample": torch.randn(2, 4, 1), "gru_cell_state": torch.randn(2, 4, 2)},
             "act": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1) / 10,
             "dec_phys_states": torch.zeros(2, 4, 25), "gt_phys_states": torch.zeros(2, 4, 25)}
    cache["dec_phys_states"][..., 1] = torch.arange(4) * 0.9
    torch.save(cache, run / "id_prior_infos.pt")
    random = {"post": {}, "prior": {name: {"states": cache["states"], "act": cache["act"],
                "start_phys": cache["gt_phys_states"][:, 0], "dec_phys": cache["dec_phys_states"]}
                for name in ("closed", "open")}}
    torch.save(random, run / "random_rollouts.pt")
    return run, config, cache, ensemble


class CacheComparisonTests(unittest.TestCase):
    """Validate offline analysis without simulator or cluster access."""

    def test_alignment_single_forward_rng_and_batching(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, config, cache, reference = fixture(Path(temporary))
            _, features, actions, _ = next(cache_groups(cache))
            model, loaded = load_ensemble(run, 3, 1, "cpu")
            self.assertEqual(loaded.log_dir, config.log_dir)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            np.testing.assert_array_equal(actions.numpy(), cache["act"].numpy())
            expected = reference(features, actions)
            state = torch.random.get_rng_state().clone()
            with patch.object(model, "forward", wraps=model.forward) as forward:
                scores, repeats, mean, std = score_inputs(model, features, actions,
                    ["gjs", "js", "transition_var"], batch_size=3, js_num_samples=16, js_repeats=3)
                self.assertEqual(forward.call_count, 3)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            np.testing.assert_allclose(mean, expected["mean"].detach().numpy(), rtol=1e-5, atol=1e-6)
            self.assertEqual(repeats.shape, (3, 2, 4))
            other = score_inputs(model, features, actions, ["gjs", "transition_var"], batch_size=8)[0]
            for name in other:
                np.testing.assert_allclose(scores[name], other[name], rtol=1e-4, atol=1e-5)
            again = score_inputs(model, features, actions, ["js"], batch_size=3, js_num_samples=16, js_repeats=3)[1]
            np.testing.assert_array_equal(repeats, again)
            cache["act"] = cache["act"][:, 1:]
            with self.assertRaises(ValueError):
                list(cache_groups(cache))

    def test_physical_mask_positions_angles(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, config, cache, _ = fixture(Path(temporary))
            error, _ = physical_discrepancy(cache, config)
            np.testing.assert_allclose(error[0], np.arange(4) * 0.1, atol=1e-7)
            # Raw x translation and all velocities are ignored; denominator stays 9.
            cache["dec_phys_states"][..., 0] = 999
            cache["dec_phys_states"][..., 16:] = 999
            same, _ = physical_discrepancy(cache, config)
            np.testing.assert_array_equal(error, same)
            cache["dec_phys_states"][..., 2] = np.sin(np.pi - 0.1)
            cache["dec_phys_states"][..., 3] = np.cos(np.pi - 0.1)
            cache["gt_phys_states"][..., 2] = np.sin(-np.pi + 0.1)
            cache["gt_phys_states"][..., 3] = np.cos(-np.pi + 0.1)
            wrapped, _ = physical_discrepancy(cache, config)
            np.testing.assert_allclose(wrapped - error, 0.2 / 9, atol=1e-7)
            config.environment.env.env = "unknown_task"
            with self.assertRaises(ValueError):
                physical_discrepancy(cache, config)

    def test_cli_immutable_sources_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _, cache, _ = fixture(root)
            before = {p: sha256(p) for p in run.iterdir()}
            output = root / "comparison"
            arguments = ["--run-dir", str(run), "--infos", str(run / "id_prior_infos.pt"),
                         str(run / "random_rollouts.pt"), "--output-dir", str(output),
                         "--batch-size", "3", "--js-num-samples", "8", "--js-repeats", "2", "--surface"]
            main(arguments)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["groups"]), 3)
            group = output / "00_id_prior_infos_prior"
            with np.load(group / "scores.npz") as scores:
                self.assertEqual(scores["gjs"].shape, (2, 4))
                self.assertEqual(scores["js_repeats"].shape, (2, 2, 4))
            self.assertTrue((group / "time_comparison.png").exists())
            with np.load(group / "trajectory_0_t_0_predictions.npz") as first, np.load(group / "trajectory_0_t_3_predictions.npz") as last:
                np.testing.assert_array_equal(first["x"], last["x"])
                np.testing.assert_array_equal(first["action"], cache["act"][0, 0])
                np.testing.assert_array_equal(last["action"], cache["act"][0, 3])
            random_group = output / "01_random_rollouts_prior_closed"
            with np.load(random_group / "scores.npz") as scores:
                self.assertNotIn("physical_discrepancy", scores.files)
            with self.assertRaises(SystemExit):
                main(arguments)
            self.assertEqual(before, {p: sha256(p) for p in run.iterdir()})

    def test_strict_weight_and_categorical_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, config, _, model = fixture(Path(temporary))
            weights = model.state_dict()
            weights.pop(next(iter(weights)))
            torch.save({"ensemble": weights}, run / "networks.pth")
            with self.assertRaises(RuntimeError):
                load_ensemble(run, 3, 1, "cpu")
            config.algorithm.name = "li-cat_urssm"
            OmegaConf.save(config, run / "config.yaml")
            with self.assertRaises(ValueError):
                load_ensemble(run, 3, 1, "cpu")

    def test_matches_original_physical_implementation(self):
        # Compile only the original pure functions, avoiding simulator imports.
        # This checks against actual repository code, not a second test formula.
        root = Path(__file__).resolve().parents[1]
        source = ast.parse((root / "uncertainty_aware_dreamer/envs/dmc/state_based_env.py").read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef))
        untransform = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "untransform_phys_state")
        source = ast.parse((root / "evaluation/strategy/broad_analysis.py").read_text())
        difference = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "compute_phys_diff")
        namespace = {"torch": torch, "np": np, "Optional": Optional,
                     "plot_funcs_shared": lambda **kwargs: None, "save_to_csv": lambda **kwargs: None}
        exec(compile(ast.Module(body=[untransform, difference], type_ignores=[]), "original_physical", "exec"), namespace)
        metadata = SuiteBaseEnv.DMC_ENV_CLASSES["cheetah"]
        base = SimpleNamespace(_base_env=SimpleNamespace(ranges=metadata.RANGES, is_angle=metadata.IS_ANGLE))
        env = SimpleNamespace(untransform_phys_state=lambda value: namespace["untransform_phys_state"](base, value),
                              get_is_angle=lambda: metadata.IS_ANGLE)
        with tempfile.TemporaryDirectory() as temporary:
            _, config, cache, _ = fixture(Path(temporary))
            cache["dec_phys_states"] = torch.randn(2, 4, 25)
            cache["gt_phys_states"] = torch.randn(2, 4, 25)
            actual, _ = physical_discrepancy(cache, config)
            expected = namespace["compute_phys_diff"](env=env,
                mask=torch.as_tensor(metadata.get_translation_inv_mask(False))[None, None],
                phys_states=[cache["dec_phys_states"]], comp_phys_states=[cache["gt_phys_states"]])[0]
            np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
