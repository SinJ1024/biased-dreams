"""Plot five Gaussian MLP predictions conditioned on exactly one state-action.

predict_fixed_input preserves model modes and returns one distribution per member.
plot_fixed_input writes analytic 2D marginal contours and optional 3D density
surfaces with all five members overlaid on shared axes, plus an NPZ archive
of the conditioning input and Gaussian parameters.
The third axis is probability density, not a third output coordinate.
"""

from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def predict_fixed_input(ensemble, state, action):
    """Evaluate singleton [features] or [1, features] inputs without state mixing."""
    parameter = next(ensemble.parameters())
    inputs = []
    for name, value in (("state", state), ("action", action)):
        value = torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[0] != 1 or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain exactly one finite input vector")
        inputs.append(value)
    modes = [(module, module.training) for module in ensemble.modules()]
    try:
        ensemble.eval()
        predictions = ensemble(state=inputs[0], action=inputs[1])
    finally:
        for module, training in modes:
            module.training = training
    mean = predictions["mean"].detach().cpu().numpy()
    std = predictions["std"].detach().cpu().numpy()
    if mean.ndim != 3 or mean.shape[:2] != (5, 1) or std.shape != mean.shape:
        raise ValueError("Expected exactly five Gaussian members with shape [5, 1, features]")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Predictions must be finite with positive standard deviations")
    return mean[:, 0], std[:, 0], [value.cpu().numpy()[0] for value in inputs]


def plot_fixed_input(ensemble, state, action, output_prefix, dimensions=(0, 1),
                     surface=False, grid_size=180, standard_deviations=4.0):
    """Save marginals of two original output coordinates for a fixed latent input.

    Contours enclose 50%, 80%, and 95% of each bivariate Gaussian's mass.
    All five members share one set of axes in each figure.
    No PCA, pooling over states, or empirical fitting is performed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    mean, std, inputs = predict_fixed_input(ensemble, state, action)
    if (len(dimensions) != 2 or len(set(dimensions)) != 2
            or any(not isinstance(d, int) or d < 0 or d >= mean.shape[-1] for d in dimensions)):
        raise ValueError("dimensions must select two distinct valid output coordinates")
    if grid_size < 32 or not np.isfinite(standard_deviations) or standard_deviations < 3:
        raise ValueError("Use grid_size >= 32 and standard_deviations >= 3")
    selected_mean, selected_std = mean[:, dimensions], std[:, dimensions]
    lower = (selected_mean - standard_deviations * selected_std).min(0)
    upper = (selected_mean + standard_deviations * selected_std).max(0)
    x, y = np.meshgrid(*(np.linspace(lower[d], upper[d], grid_size) for d in range(2)))
    # Marginalizing a diagonal Gaussian simply selects means and variances.
    peak = 1.0 / (2.0 * np.pi * selected_std.prod(-1))
    density = peak[:, None, None] * np.exp(-0.5 * (
        ((x[None] - selected_mean[:, 0, None, None]) / selected_std[:, 0, None, None]) ** 2
        + ((y[None] - selected_mean[:, 1, None, None]) / selected_std[:, 1, None, None]) ** 2))
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    xlabel, ylabel = (f"Output dimension {dimension}" for dimension in dimensions)
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for member in range(5):
        axis.contour(x, y, density[member], levels=peak[member] * np.array([0.05, 0.2, 0.5]),
                     colors=[colors[member]], linewidths=1.6)
        axis.plot(*selected_mean[member], marker="+", color=colors[member])
    axis.set(xlabel=xlabel, ylabel=ylabel, xlim=(lower[0], upper[0]),
             ylim=(lower[1], upper[1]))
    axis.legend(handles=[Line2D([0], [0], color=color, label=f"MLP {i + 1}")
                         for i, color in enumerate(colors)])
    fig.suptitle("One fixed state-action: 50%, 80%, 95% marginal probability contours")
    contour_path = str(prefix) + "_contours.png"
    fig.savefig(contour_path, dpi=180)
    plt.close(fig)
    paths = [contour_path]
    if surface:
        fig = plt.figure(figsize=(11, 9), constrained_layout=True)
        axis = fig.add_subplot(111, projection="3d")
        for member in range(5):
            # Hide near-zero tails only in the surface rendering to avoid five
            # overlapping floor sheets. The archived densities remain complete.
            visible_density = np.where(density[member] >= peak[member] * 0.01,
                                       density[member], np.nan)
            axis.plot_surface(x, y, visible_density, color=colors[member], alpha=0.35,
                              linewidth=0, antialiased=True,
                              rcount=min(grid_size, 100), ccount=min(grid_size, 100))
            axis.contour(x, y, density[member], levels=[peak[member] * 0.05],
                         zdir="z", offset=0, colors=[colors[member]], linewidths=1.2)
        axis.set(xlabel=xlabel, ylabel=ylabel, zlabel="Probability density",
                 xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                 zlim=(0, float(peak.max()) * 1.05))
        axis.view_init(elev=28, azim=-55)
        axis.legend(handles=[Patch(facecolor=color, alpha=0.35, label=f"MLP {i + 1}")
                             for i, color in enumerate(colors)], loc="upper right")
        fig.suptitle("One fixed state-action: five Gaussian density surfaces")
        surface_path = str(prefix) + "_surfaces.png"
        fig.savefig(surface_path, dpi=180)
        plt.close(fig)
        paths.append(surface_path)
    archive_path = str(prefix) + "_predictions.npz"
    np.savez(archive_path, state=inputs[0], action=inputs[1], mean=mean, std=std,
             dimensions=np.asarray(dimensions), x=x, y=y, density=density)
    return paths + [archive_path]
