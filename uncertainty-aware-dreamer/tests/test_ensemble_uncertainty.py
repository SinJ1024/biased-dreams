"""Verify disagreement semantics, legacy weight loading, and fixed-input plots.

Unittest cases cover analytic limits, numerical integration, RNG behavior,
training gradients, configuration choices, and rejection of mixed-state plots.
Run from uncertainty-aware-dreamer with python -m unittest discover -s tests.
"""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.ensemble_distribution import plot_fixed_input, predict_fixed_input
from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.distance_measure import (
    GeometricJensenShannonDivergence, JensenShannonDivergence, TransitionMeanVariance)
from uncertainty_aware_dreamer.ssm_mbrl.uncertainty.ensemble import EnsembleModel
from uncertainty_aware_dreamer.ssm_mbrl.util.config_dict import ConfigDict


def make_ensemble(measure="gjs"):
    """Construct a small real ensemble with legacy configuration fields only."""
    config = ConfigDict(target="deter", num_ensemble=5, hidden_size=8,
                        activation="ELU", normalization="LayerNorm", num_layers=2,
                        init_std=1.0, min_std=0.1, max_std=5.0,
                        sigmoid_activation=False, output_normalization="none",
                        distance_measure=measure)
    return EnsembleModel(3, 2, 2, 2, 1, False, config)


class DisagreementTests(unittest.TestCase):
    """Test mathematical meaning independently of the training implementation."""

    def setUp(self):
        torch.manual_seed(7)

    def test_identical_and_separated_js(self):
        means = torch.zeros(5, 2, 3, dtype=torch.float64)
        variance = torch.ones_like(means)
        js = JensenShannonDivergence(128, 16)
        torch.testing.assert_close(js.compute_measure(means, variance), torch.zeros(2, 1, dtype=means.dtype))
        means[:, :, 0] = torch.arange(5, dtype=means.dtype)[:, None] * 30
        torch.testing.assert_close(js.compute_measure(means, variance),
                                   torch.full((2, 1), math.log(5), dtype=means.dtype))

    def test_js_matches_numerical_integral(self):
        means = torch.tensor([[[-0.7]], [[0.9]]], dtype=torch.float64)
        variance = torch.tensor([[[0.5]], [[1.4]]], dtype=torch.float64)
        x = torch.linspace(-12, 12, 30001, dtype=means.dtype)
        p = torch.exp(-0.5 * (x - means[:, 0]) ** 2 / variance[:, 0]) / (2 * math.pi * variance[:, 0]).sqrt()
        reference = torch.trapezoid(p * (p.log() - p.mean(0).log()), x, dim=-1).mean()
        estimate = JensenShannonDivergence(20000, 512).compute_measure(means, variance).item()
        self.assertAlmostEqual(estimate, reference.item(), delta=0.012)

    def test_mean_variance_ignores_predictive_spread(self):
        means = torch.tensor([[0., 2.], [2., 6.]]).reshape(2, 1, 2)
        metric = TransitionMeanVariance()
        self.assertEqual(metric.compute_measure(means, torch.ones_like(means)).item(), 2.5)
        self.assertEqual(metric.compute_measure(means, torch.ones_like(means) * 100).item(), 2.5)
        means.zero_()
        variance = torch.tensor([0.1, 10.]).reshape(2, 1, 1).expand_as(means)
        self.assertEqual(metric.compute_measure(means, variance).item(), 0)
        self.assertGreater(JensenShannonDivergence(2000).compute_measure(means, variance).item(), 0.1)

    def test_shapes_single_member_and_rng(self):
        for shape in ((5, 4), (5, 2, 4), (5, 2, 3, 4)):
            means = torch.randn(shape)
            for metric in (JensenShannonDivergence(), TransitionMeanVariance()):
                self.assertEqual(metric.compute_measure(means, torch.ones_like(means)).shape, shape[1:-1] + (1,))
        means = torch.randn(1, 2, 4)
        self.assertEqual(JensenShannonDivergence().compute_measure(means, torch.ones_like(means)).sum(), 0)
        metric = JensenShannonDivergence()
        means = torch.randn(5, 2, 4)
        torch.manual_seed(42)
        first = metric.compute_measure(means, torch.ones_like(means))
        torch.manual_seed(42)
        torch.testing.assert_close(first, metric.compute_measure(means, torch.ones_like(means)), rtol=0, atol=0)
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                JensenShannonDivergence(count)

    def test_legacy_gjs_analytic_case_and_rng(self):
        means = torch.tensor([[[0.]], [[2.]]])
        rng = torch.random.get_rng_state()
        self.assertEqual(GeometricJensenShannonDivergence().compute_measure(means, torch.ones_like(means)).item(), 0.5)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_checkpoint_and_training_compatibility(self):
        original = make_ensemble()
        legacy_weights = original.state_dict()
        self.assertFalse(any(key.startswith("_distance_measure") for key in legacy_weights))
        states, actions = torch.randn(2, 3, 3), torch.randn(2, 3, 1)
        expected = original(states, actions)
        for measure in ("gjs", "js", "transition_var", "jrd"):
            ensemble = make_ensemble(measure)
            ensemble.load_state_dict(legacy_weights, strict=True)
            self.assertEqual(set(ensemble.state_dict()), set(legacy_weights))
            torch.testing.assert_close(ensemble(states, actions)["mean"], expected["mean"], rtol=0, atol=0)
            torch.testing.assert_close(ensemble(states, actions)["std"], expected["std"], rtol=0, atol=0)
            self.assertEqual(ensemble.compute_disagreement(states, actions).shape, (2, 3, 1))
            loss, _ = ensemble.compute_loss(states, actions, torch.randn(2, 3, 2))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in ensemble.parameters()))

    def test_plot_rejects_mixed_states_and_restores_modes(self):
        ensemble = make_ensemble()
        ensemble.train()
        ensemble._models[0].eval()
        modes = [module.training for module in ensemble.modules()]
        with self.assertRaises(ValueError):
            predict_fixed_input(ensemble, torch.zeros(2, 3), torch.zeros(2, 1))
        mean, std, _ = predict_fixed_input(ensemble, torch.zeros(3), torch.zeros(1))
        self.assertEqual(mean.shape, (5, 2))
        self.assertEqual([module.training for module in ensemble.modules()], modes)
        with tempfile.TemporaryDirectory() as directory:
            paths = plot_fixed_input(ensemble, torch.zeros(3), torch.zeros(1),
                                     Path(directory) / "fixed", surface=True)
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in paths))
            with np.load(paths[-1]) as data:
                np.testing.assert_array_equal(data["mean"], mean)
                np.testing.assert_array_equal(data["std"], std)
                density = data["density"]
                # Explicit trapezoids work with both NumPy 1.26 and NumPy 2.x.
                row_mass = ((density[..., 1:] + density[..., :-1]) * 0.5 * np.diff(data["x"][0])).sum(-1)
                integral = ((row_mass[:, 1:] + row_mass[:, :-1]) * 0.5 * np.diff(data["y"][:, 0])).sum(-1)
                np.testing.assert_allclose(integral, np.ones(5), atol=0.002)

    def test_saved_checkpoint_plot_cli(self):
        ensemble = make_ensemble()
        config = dict(target="deter", num_ensemble=5, hidden_size=8,
                      activation="ELU", normalization="LayerNorm", num_layers=2,
                      init_std=1.0, min_std=0.1, max_std=5.0,
                      sigmoid_activation=False, output_normalization="none", distance_measure="gjs")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            OmegaConf.save(OmegaConf.create({"algorithm": {"ensemble": config}}), run / "config.yaml")
            torch.save({"ensemble": ensemble.state_dict()}, run / "networks.pth")
            np.savez(run / "input.npz", state=np.zeros(3), action=np.zeros(1))
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "plot_ensemble.py"),
                                        "--run-dir", str(run), "--input", str(run / "input.npz"),
                                        "--surface", "--output-prefix", str(run / "cli")],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((run / "cli_surfaces.png").exists())


if __name__ == "__main__":
    unittest.main()
