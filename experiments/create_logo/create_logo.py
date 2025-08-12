import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib import transforms
import random


def _smooth_noise(x_len, scale=0.08, kernel_sigma=6):
    """
    Create smooth 1D noise of length x_len.
    scale: noise amplitude
    kernel_sigma: controls smoothness (larger = smoother)
    """
    # White noise
    n = np.random.normal(0.0, 1.0, x_len)
    # Make a Gaussian kernel
    kx = np.arange(-3 * kernel_sigma, 3 * kernel_sigma + 1)
    kernel = np.exp(-(kx**2) / (2 * kernel_sigma**2))
    kernel /= kernel.sum()
    # Convolve to get smooth noise
    smooth = np.convolve(n, kernel, mode="same")
    # Normalize and scale
    smooth = smooth / (np.std(smooth) + 1e-8)
    return scale * smooth


def create_logo(
    text="FROST",
    size_px=768,
    dpi=192,
    seed=7,
    save_png_path=None,
    save_svg_path=None,
    show=False,
    text_scale_x=1.0,
):
    """
    Generate a polished logo with soft gradients, a banner, layered organic curves (multi-color), and glow.

    Parameters:
      - text: main logo text
      - size_px: canvas size (square) in pixels
      - dpi: figure DPI
      - seed: randomness seed for consistent curves
      - save_png_path: path to save PNG (transparent bg) if provided
      - save_svg_path: path to save SVG if provided
      - show: whether to display interactively
      - text_scale_x: horizontal scale for the text (e.g., 1.2 to stretch, 0.9 to compress)
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Palette (cool spectrum with extra variety)
    base_light = "#DFFFFF"
    base_cyan = "#00AEEF"
    banner_dark = "#0C6C9A"
    banner_light = "#3FC3F6"

    # Multihue curve palette (teal, cyan, blue, indigo, purple, aqua)
    curve_colors = [
        "#0A4A6A", "#117C88", "#29BEB8", "#66E0C2",
        "#0077CC", "#0D79A7", "#00AEEF", "#3FC3F6",
        "#5ED6FF", "#3F71C6", "#6A5ACD", "#7A9BFF",
        "#8A7FFF", "#B08CFF", "#9FD3FF"
    ]
    curve_cmap = LinearSegmentedColormap.from_list("frost_multi", curve_colors, N=512)

    # Figure setup
    inches = size_px / dpi
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    ax.set_facecolor("none")
    ax.axis("off")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Radial background glow clipped to circle (no visible circle outline)
    res = 512
    yy, xx = np.mgrid[0:1:complex(res), 0:1:complex(res)]
    cx, cy = 0.5, 0.5
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / 0.48
    radial = np.clip(1 - r, 0, 1)
    bg_cmap = LinearSegmentedColormap.from_list("frost_bg", [base_light, "#EFFFFF", "#F7FFFF"])
    bg_img = bg_cmap(0.75 + 0.25 * radial)

    # Use circle as clip path only (no stroke)
    circle = Circle((0.5, 0.5), 0.48, edgecolor="none", facecolor="none", lw=0, zorder=2)
    ax.add_patch(circle)
    ax.imshow(
        bg_img,
        extent=(0, 1, 0, 1),
        origin="lower",
        interpolation="bilinear",
        zorder=1,
        clip_path=circle,
        clip_on=True,
    )

    # Soft outer glow rings (very subtle)
    for i, alpha in enumerate([0.08, 0.05, 0.03, 0.015]):
        glow = Circle((0.5, 0.5), 0.48 + 0.008 * (i + 1), facecolor=to_rgba(base_cyan, alpha), edgecolor="none", zorder=0)
        ax.add_patch(glow)

    # Banner geometry (keep rounded shape but remove its outline)
    banner_h = 0.18
    banner_y0 = 0.5 - banner_h / 2
    banner_radius = 0.07

    # Banner gradient (horizontal)
    grad_w = 1024
    grad = np.linspace(0, 1, grad_w)[None, :]
    banner_cmap = LinearSegmentedColormap.from_list("banner", [banner_dark, banner_light])
    banner_img = banner_cmap(grad)

    banner_patch = FancyBboxPatch(
        (0.08, banner_y0),
        0.84,
        banner_h,
        boxstyle=f"round,pad=0.015,rounding_size={banner_radius}",
        facecolor="none",
        edgecolor="none",  # removed rounded outline
        linewidth=0,
        zorder=4,
    )
    ax.add_patch(banner_patch)

    ax.imshow(
        banner_img,
        extent=(0.08, 0.92, banner_y0, banner_y0 + banner_h),
        origin="lower",
        interpolation="bicubic",
        zorder=3,
        clip_path=banner_patch,
        clip_on=True,
    )

    # Subtle banner inner highlight
    highlight = np.clip(1.0 - ((yy - (banner_y0 + banner_h * 0.65)) / (banner_h * 0.8)) ** 2, 0, 1) ** 2
    highlight = (highlight * 0.08)[..., None]
    highlight_img = np.ones((res, res, 4))
    highlight_img[..., :3] = 1.0
    highlight_img[..., 3] = highlight[..., 0]
    ax.imshow(
        highlight_img,
        extent=(0, 1, 0, 1),
        origin="lower",
        interpolation="bilinear",
        zorder=4,
        clip_path=banner_patch,
        clip_on=True,
    )

    # Curves: more random, organic, multi-color
    upper_zero_line = banner_y0
    lower_zero_line = banner_y0 + banner_h

    x = np.linspace(-1, 1, 800)
    x_plot = 0.5 + x * 0.45

    # Randomized centers, amplitudes, widths
    n_curves = 13
    centers = np.linspace(-0.38, 0.38, n_curves) + np.random.normal(0.0, 0.04, n_curves)
    amps = np.clip(np.random.lognormal(mean=np.log(0.22), sigma=0.35, size=n_curves), 0.10, 0.40)
    widths = np.random.uniform(7.0, 14.0, size=n_curves)  # larger = sharper

    # Random color positions across the multihue colormap
    color_positions = np.clip(np.sort(np.random.uniform(0.15, 0.95, n_curves)), 0, 1)
    colors = [curve_cmap(p) for p in color_positions]

    # Slight per-curve alpha and linewidth variation
    stroke_lws = np.random.uniform(1.2, 1.9, n_curves)
    fill_alphas = np.random.uniform(0.22, 0.36, n_curves)

    # Precompute smooth noise to add gentle waviness
    noise = _smooth_noise(x.size, scale=0.08, kernel_sigma=10)

    for idx, (c, a, w) in enumerate(zip(centers, amps, widths)):
        base = np.exp(-w * (x - c) ** 2)
        y = a * base * (1.0 + 0.15 * noise)  # small organic variation

        # Upper fill and stroke
        ax.fill_between(
            x_plot,
            upper_zero_line,
            upper_zero_line - y,
            color=to_rgba(colors[idx], fill_alphas[idx]),
            zorder=2.5,
            linewidth=0,
            antialiased=True,
        )
        ax.plot(
            x_plot,
            upper_zero_line - y,
            color=to_rgba(colors[idx], 0.88),
            lw=stroke_lws[idx],
            zorder=3.0,
            solid_capstyle="round",
            antialiased=True,
        )

        # Lower fill and stroke
        ax.fill_between(
            x_plot,
            lower_zero_line,
            lower_zero_line + y,
            color=to_rgba(colors[idx], fill_alphas[idx]),
            zorder=2.5,
            linewidth=0,
            antialiased=True,
        )
        ax.plot(
            x_plot,
            lower_zero_line + y,
            color=to_rgba(colors[idx], 0.88),
            lw=stroke_lws[idx],
            zorder=3.0,
            solid_capstyle="round",
            antialiased=True,
        )

    # Text transform to stretch around center (0.5, 0.5) in Axes coords
    text_transform = transforms.Affine2D().translate(-0.5, -0.5).scale(text_scale_x, 1.0).translate(0.5, 0.5) + ax.transAxes

    # Text styling
    text_artist = ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=44,
        color="white",
        weight="bold",
        zorder=6,
        fontfamily="DejaVu Sans",
        transform=text_transform,
    )
    text_artist.set_path_effects(
        [
            path_effects.Stroke(linewidth=2.2, foreground="#0A3B54"),
            path_effects.withSimplePatchShadow(offset=(0.6, -0.6), shadow_rgbFace=(0, 0, 0), alpha=0.25),
            path_effects.Normal(),
        ]
    )

    # Thin top highlight arc inside circle (kept subtle)
    arc_res = 512
    theta = np.linspace(np.pi + 0.25, 2 * np.pi - 0.25, arc_res)
    arc_r = 0.47
    arc_x = 0.5 + arc_r * np.cos(theta)
    arc_y = 0.5 + arc_r * np.sin(theta)
    ax.plot(arc_x, arc_y, color=to_rgba("#FFFFFF", 0.28), lw=1.1, zorder=5, solid_capstyle="round")

    plt.tight_layout(pad=0)

    if save_png_path:
        fig.savefig(save_png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="none", transparent=True)
    if save_svg_path:
        fig.savefig(save_svg_path, bbox_inches="tight", pad_inches=0.02, facecolor="none", transparent=True)

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # Example usage
    create_logo(
        text="FROST",
        size_px=768,
        dpi=192,
        seed=11,           # tweak seed for different random curves/colors
        text_scale_x=1.2,  # horizontal stretch
        save_png_path="logo.png",
        save_svg_path="logo.svg",
        show=False,
    )
