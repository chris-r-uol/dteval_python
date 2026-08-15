"""Python-native figures for the DTEval port.

These are deliberately NOT reproductions of the R package's ggplot2 output.
The parity contract is about *values*, and those are gated separately; the
presentation here is matplotlib's own idiom -- perceptually uniform colour
maps, masked surfaces, hexbin where a scatter would overplot.

The spatial figures all mask the fitted surface with `too_far`, so the map
shows a hole where there are no tubes rather than an extrapolation. That is the
difference between a map of measurements and a map of guesses.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import dteval as dte  # noqa: E402
from dteval.plots.basemap import attribute, fetch_basemap, to_mercator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "figures" / "native"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, RULE = "#141C18", "#5B6B63", "#DBE2DD"
CONC_CMAP = "magma"
CAZ_LINE = "#22D3A6"

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    "axes.titlelocation": "left",
    "axes.titlepad": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
})


# ---------------------------------------------------------------- helpers ---
def caz_rings() -> list[np.ndarray]:
    """The Bradford CAZ boundary, projected to Mercator metres."""
    geom = dte.datasets.caz_brd().geoms[0]
    parts = getattr(geom, "geoms", [geom])
    rings = []
    for part in parts:
        coords = np.asarray(part.coords)
        x, y = to_mercator(coords[:, 0], coords[:, 1])
        rings.append(np.column_stack([x, y]))
    return rings


def add_basemap(ax, bounds, zoom=None):
    """Draw the OSM basemap and pin the axes to the data's own extent."""
    image, extent = fetch_basemap(bounds, zoom=zoom)
    ax.imshow(image, extent=extent, origin="upper", interpolation="bilinear", zorder=0)
    west, south, east, north = bounds
    (x0, x1), (y0, y1) = to_mercator([west, east], [south, north])
    pad_x, pad_y = (x1 - x0) * 0.02, (y1 - y0) * 0.02
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_aspect("equal")  # both axes are metres now, so this is simply true
    attribute(ax)


def surface_grid(frame: pd.DataFrame):
    """Reshape a fitted surface into its grid, projected to Mercator metres.

    Mercator stretches latitude non-linearly, so the projected row coordinates
    are not evenly spaced. pcolormesh takes coordinate arrays rather than an
    extent, so passing them through keeps the surface aligned with the tiles
    exactly instead of approximately.
    """
    lon = np.sort(frame[".longitude"].unique())
    lat = np.sort(frame[".latitude"].unique())
    pivot = frame.pivot_table(
        index=".latitude", columns=".longitude", values=".value.pred", dropna=False
    ).reindex(index=lat, columns=lon)
    x, _ = to_mercator(lon, np.zeros_like(lon))
    _, y = to_mercator(np.zeros_like(lat), lat)
    return x, y, np.ma.masked_invalid(pivot.to_numpy())


def data_bounds(pad: float = 0.012):
    """Bounding box of the tube network, in degrees."""
    d = dte.tag_tube(dte.datasets.dt_brd())
    lon, lat = d[".longitude"].dropna(), d[".latitude"].dropna()
    return (lon.min() - pad, lat.min() - pad * 0.6, lon.max() + pad, lat.max() + pad * 0.6)


def draw_caz(ax, label: bool = False, lw: float = 1.6):
    for i, ring in enumerate(caz_rings()):
        ax.plot(ring[:, 0], ring[:, 1], color=CAZ_LINE, lw=lw,
                solid_joinstyle="round", zorder=5,
                label="Clean Air Zone" if (label and i == 0) else None)


def titled(ax, title: str, subtitle: str | None = None, size: float = 11.5):
    """Title with an optional deck beneath it.

    The deck sits just above the axes, so the title has to be padded clear of
    it -- otherwise matplotlib draws both at roughly the same height and they
    overlap.
    """
    pad = 22 if subtitle else 9
    ax.set_title(title, pad=pad, fontsize=size)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, va="bottom", ha="left",
                color=MUTED, fontsize=9)


def strip_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ------------------------------------------------------------- 1. surface ---
def fig_surface() -> str:
    grid = dte.fit_tube_model(
        dte.datasets.dt_brd(), tube=".value", inputs=[".longitude", ".latitude"],
        new_data="input.ranges", simplify=True, grid_resolution=220, too_far=0.055,
    )
    x, y, z = surface_grid(grid)
    sites = (
        dte.tag_tube(dte.datasets.dt_brd())
        .groupby([".longitude", ".latitude"], observed=True)[".value"]
        .mean().reset_index()
    )
    site_x, site_y = to_mercator(sites[".longitude"], sites[".latitude"])

    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    add_basemap(ax, data_bounds())
    mesh = ax.pcolormesh(x, y, z, cmap=CONC_CMAP, shading="gouraud", alpha=0.74, zorder=2,
                         norm=Normalize(vmin=np.floor(z.min()), vmax=np.ceil(z.max())))
    ax.contour(x, y, z, levels=8, colors="white", linewidths=0.4, alpha=0.4, zorder=3)
    draw_caz(ax, label=True)
    ax.scatter(site_x, site_y, s=8, c="white", edgecolors="black", linewidths=0.4,
               alpha=0.9, zorder=6, label=f"{len(sites)} tube locations")
    strip_axes(ax)
    titled(ax, "Modelled NO$_2$ across Bradford",
           "LOESS surface over all sampling periods, 2022–2026", size=13)

    bar = fig.colorbar(mesh, ax=ax, fraction=0.041, pad=0.02)
    bar.set_label("NO$_2$  (µg m$^{-3}$)", color=INK)
    bar.outline.set_visible(False)
    ax.legend(loc="lower left", fontsize=8, labelcolor=INK,
              handletextpad=0.5, borderpad=0.7)
    ax.text(0, -0.025, "Where the surface is absent, no tube lies within the fit's "
                       "reach; the basemap shows through rather than an extrapolation.",
            transform=ax.transAxes, va="top", ha="left", color=MUTED, fontsize=8)

    path = OUT / "surface.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path.name


# ------------------------------------------------------- 2. surface/year ----
def fig_surface_years() -> str:
    grid = dte.fit_tube_model(
        dte.datasets.dt_brd(), tube=".value", inputs=[".longitude", ".latitude"],
        by=".year", new_data="input.ranges", simplify=True,
        grid_resolution=150, too_far=0.05,
    )
    years = sorted(grid[".year"].dropna().unique())
    vmin = np.floor(grid[".value.pred"].min())
    vmax = np.ceil(grid[".value.pred"].max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    bounds = data_bounds()
    fig, axes = plt.subplots(1, len(years), figsize=(3.1 * len(years), 4.2))
    for ax, year in zip(np.atleast_1d(axes), years, strict=True):
        x, y, z = surface_grid(grid[grid[".year"] == year])
        add_basemap(ax, bounds)
        mesh = ax.pcolormesh(x, y, z, cmap=CONC_CMAP, shading="gouraud",
                             norm=norm, alpha=0.76, zorder=2)
        draw_caz(ax, lw=1.1)
        strip_axes(ax)
        median = np.ma.median(z)
        ax.set_title(f"{year}", fontsize=11)
        ax.text(0.5, -0.03, f"median {median:.1f}", transform=ax.transAxes,
                ha="center", va="top", color=MUTED, fontsize=8.5)

    # 26.7 -> 23.9 -> 21.7 -> 23.9: a fall to 2024 and a partial rebound, not a
    # steady decline. The title says what the numbers say.
    fig.suptitle("The city-centre hotspot holds its position year on year",
                 x=0.008, y=1.16, ha="left", fontsize=12.5, weight="medium")
    fig.text(0.008, 1.06, "Surface median falls to 2024, then rebounds",
             ha="left", color=MUTED, fontsize=9)
    bar = fig.colorbar(mesh, ax=list(np.atleast_1d(axes)), fraction=0.022, pad=0.015)
    bar.set_label("NO$_2$  (µg m$^{-3}$)", color=INK)
    bar.outline.set_visible(False)

    path = OUT / "surface-years.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path.name


# ------------------------------------------------------------ 3. CAZ split --
def fig_caz() -> str:
    tagged = dte.tube_in_xy_polygon(dte.datasets.dt_brd(), polygon=dte.datasets.caz_brd())
    tagged = dte.tag_tube(tagged)
    inside = tagged[tagged[".in_polygon"]][".value"].dropna()
    outside = tagged[~tagged[".in_polygon"]][".value"].dropna()

    fig, (ax_map, ax_dist) = plt.subplots(
        1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1, 1.5]}
    )

    sites = tagged.groupby(
        [".longitude", ".latitude"], observed=True
    ).agg(inside=(".in_polygon", "first"), value=(".value", "mean")).reset_index()
    add_basemap(ax_map, data_bounds())
    for flag, colour, label in ((True, "#C2410C", "inside"), (False, "#334155", "outside")):
        part = sites[sites["inside"] == flag]
        px, py = to_mercator(part[".longitude"], part[".latitude"])
        ax_map.scatter(px, py, s=18, c=colour, alpha=0.9, linewidths=0.4,
                       edgecolors="white", zorder=6, label=f"{label} ({len(part)})")
    draw_caz(ax_map, lw=1.4)
    strip_axes(ax_map)
    titled(ax_map, "Tube sites by zone", size=10.5)
    ax_map.legend(loc="upper left", fontsize=8, labelcolor=INK)

    bins = np.linspace(0, max(inside.max(), outside.max()), 46)
    for values, colour, label in (
        (outside, "#4C5A53", f"outside  n={len(outside):,}"),
        (inside, "#B02E0C", f"inside  n={len(inside):,}"),
    ):
        ax_dist.hist(values, bins=bins, density=True, histtype="stepfilled",
                     alpha=0.45, color=colour, label=label)
        ax_dist.axvline(values.median(), color=colour, lw=1.4, ls="--")
    ax_dist.set_xlabel("NO$_2$  (µg m$^{-3}$)")
    ax_dist.set_ylabel("density")
    titled(ax_dist, "Concentration distribution", size=10.5)
    ax_dist.legend(fontsize=8, labelcolor=INK)
    ax_dist.grid(axis="y", alpha=0.5)
    ax_dist.set_axisbelow(True)
    ax_dist.text(
        0.98, 0.62,
        f"median {inside.median():.1f} inside\nagainst {outside.median():.1f} outside",
        transform=ax_dist.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED,
    )

    fig.suptitle("Inside the Clean Air Zone against outside it",
                 x=0.008, y=1.06, ha="left", fontsize=12.5, weight="medium")
    path = OUT / "caz.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path.name


# ------------------------------------------------------------ 4. precision --
def fig_precision() -> str:
    res = dte.test_tube_precision(dte.datasets.dt_brd(), show=[])
    d = res["data"]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    hb = ax.hexbin(d[".mean"], d[".value"], gridsize=58, bins="log",
                   cmap="magma_r", mincnt=1, linewidths=0)
    lo, hi = d[".mean"].min(), d[".mean"].max()
    ax.plot([lo, hi], [lo, hi], color="#0E6B4C", lw=1.2, ls="--", zorder=4)

    fit = d.sort_values(".mean")
    for column, style in ((".y", "-"), (".ylow", ":"), (".yhigh", ":")):
        if column in fit:
            ax.plot(fit[".mean"], fit[column], color="#0E6B4C", lw=1.4, ls=style, zorder=5)

    ax.set_xlabel("replicate mean  (µg m$^{-3}$)")
    ax.set_ylabel("individual tube  (µg m$^{-3}$)")
    titled(ax, "Replicate precision",
           f"{len(d):,} tubes in co-located sets, against their own set mean")
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], color="#0E6B4C", ls="--", label="1:1"),
        Line2D([], [], color="#0E6B4C", label="LOESS fit"),
        Line2D([], [], color="#0E6B4C", ls=":", label="precision bounds"),
    ], loc="upper left", fontsize=8, labelcolor=INK)

    bar = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("tubes per cell", color=INK)
    bar.outline.set_visible(False)
    path = OUT / "precision-hex.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path.name


# ------------------------------------------------------------- 5. seasonal --
def fig_seasonal() -> str:
    d = dte.tag_tube_required(dte.datasets.dt_brd(), required=[".value", ".month"])
    d = d.dropna(subset=[".value", ".month"])
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    groups = [d[d[".month"].astype(str) == m][".value"].to_numpy() for m in order]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    parts = ax.violinplot(groups, positions=range(12), widths=0.85,
                          showextrema=False, showmedians=False)
    cmap = plt.get_cmap(CONC_CMAP)
    medians = np.array([np.median(g) if len(g) else np.nan for g in groups])
    norm = Normalize(np.nanmin(medians), np.nanmax(medians))
    for body, median in zip(parts["bodies"], medians, strict=True):
        # magma bottoms out at near-black, which reads as heavy rather than low
        # next to the maps; keep to the upper part of the ramp.
        body.set_facecolor(cmap(0.3 + 0.62 * norm(median)))
        body.set_alpha(0.9)
        body.set_linewidth(0)
    ax.plot(range(12), medians, color=INK, lw=1.2, marker="o", ms=3.5, zorder=5)

    ax.set_xticks(range(12), order)
    ax.set_ylabel("NO$_2$  (µg m$^{-3}$)")
    titled(ax, "Seasonal cycle",
           f"every tube reading by calendar month; winter median "
           f"{np.nanmax(medians):.0f} against {np.nanmin(medians):.0f} in summer")
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    path = OUT / "seasonal.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path.name


if __name__ == "__main__":
    for fn in (fig_surface, fig_surface_years, fig_caz, fig_precision, fig_seasonal):
        print(f"  {fn.__name__:<20} -> {fn()}", flush=True)
    print(f"\nwrote 5 figure(s) to {OUT}")
