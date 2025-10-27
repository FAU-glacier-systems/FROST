import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


cmap = plt.get_cmap("tab20").colors
colors = tuple(cmap[i] for i in range(0, 20, 2)) + tuple(
    cmap[i] for i in range(1, 20, 2))

def scatter_plot(ax, x, y, xlabel, ylabel, title, ticks, glacier_names=None,
                 y_std=None, x_std=None, color=None):
    margin = (ticks[-1] - ticks[0]) * 0.05
    x_min = ticks[0] - margin
    x_max = ticks[-1] + margin

    ax.plot([x_min, x_max], [x_min, x_max], "--", color="black", alpha=0.3,
            zorder=-4,
            label="1:1 Correlation")

    # y = predictions, x = observations
    mae = np.mean(np.abs(y - x))
    bias = np.mean(y - x)
    y_corr = y - bias
    bc_didf = y_corr - x
    bcmae = np.mean(np.abs(bc_didf))  # Bias-corrected MAE
    correlation = np.corrcoef(x, y)[0, 1]
    txt = (
        f"Correlation: {correlation:.2f}\n"
        f"Mean error: {mae:.2f}"

    )
    ax.text(0.95, 0.05,txt,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='bottom', zorder=100,
            horizontalalignment='right')

    scatter_handles = []
    if glacier_names is None:
        scatter = ax.scatter(x, y)
    else:
        for i, label in enumerate(glacier_names):
            scatter = ax.scatter(x[i], y[i], label=label,
                                 color=colors[i % len(colors)], zorder=10)
            scatter_handles.append(scatter)
            if x_std is not None and y_std is not None:
                ellipse = Ellipse(
                    (x[i], y[i]),
                    width=2 * x_std[i],
                    height=2 * y_std[i],
                    edgecolor='none', facecolor=colors[i % len(colors)],
                    alpha=0.1, zorder=2
                )
                ax.add_patch(ellipse)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis="y", color="lightgray", linestyle="-", zorder=-10)
    ax.grid(axis="x", color="lightgray", linestyle="-", zorder=-10)
    ax.xaxis.set_tick_params(bottom=False)
    ax.yaxis.set_tick_params(left=False)


    return scatter_handles


