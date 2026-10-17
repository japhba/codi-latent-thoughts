"""CODI's continuous feedback channel, in a layer-by-position grid (post: CODI section).

CPU, matplotlib. Output: figs/codi_flow.png (env FIG_DIR).
"""
import os, pathlib, sys
from math import hypot, sqrt

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from codi_repro.analysis import ROOT, read  # noqa: E402

COLORS = {'attention': '0.65', 'path': '#008577', 'feedback': '#d97816', 'token': '#3178b5'}
# Tiny column vectors beside the payload nodes: one-hot text tokens vs continuous thoughts.
GLYPH_HOT = [1, 3]
GLYPH_VALUES = [[.25, .9, .5, .1, .65], [.75, .2, .95, .4, .55]]


def arrow(ax, source, target, color):
    ax.annotate('', xy=target.center, xytext=source.center, arrowprops={'patchA': source, 'patchB': target, 'arrowstyle': '->', 'color': color}, zorder=2)


def feedback(ax, source, target, color):
    """Exit upward, turn right 180 degrees, descend, then turn left 180 degrees."""
    sx, sy = source.center
    tx, ty = target.center
    track, upper, lower, turn = sx + 1.25, sy + source.radius + .4, ty - target.radius - .4, .6
    top_mid, bottom_mid = (sx + track) / 2, (track + tx) / 2
    top_radius, bottom_radius = (track - sx) / 2, (tx - track) / 2
    k = 4 * (sqrt(2) - 1) / 3
    vertices = [(sx, sy + source.radius), (sx, upper),
                (sx, upper + k * turn), (top_mid - k * top_radius, upper + turn), (top_mid, upper + turn),
                (top_mid + k * top_radius, upper + turn), (track, upper + k * turn), (track, upper),
                (track, lower),
                (track, lower - k * turn), (bottom_mid - k * bottom_radius, lower - turn), (bottom_mid, lower - turn),
                (bottom_mid + k * bottom_radius, lower - turn), (tx, lower - k * turn), (tx, lower),
                (tx, ty - target.radius)]
    codes = [Path.MOVETO, Path.LINETO, *([Path.CURVE4] * 6), Path.LINETO, *([Path.CURVE4] * 6), Path.LINETO]
    ax.add_patch(FancyArrowPatch(path=Path(vertices, codes), arrowstyle='->', mutation_scale=plt.rcParams['font.size'], color=color, zorder=2))


def reproduce(out, root=ROOT):
    out.mkdir(parents=True, exist_ok=True)
    manifest = read('manifest.json', root)
    layers, thoughts = read('artifacts/base_config.json', root)['num_hidden_layers'], manifest['num_latent']
    assert layers == 36 and thoughts == 6
    fig, axes = plt.subplots(1, 2, layout='constrained')
    panels, texts, glyphs = [], [], []
    for ax, mode, title in zip(axes, ['tokens', 'latent'], ['Conventional CoT', 'Latent CoT']):
        positions = [0, 2.5, 6] if mode == 'tokens' else [0, 2.5, 6, 8.5]
        indices = ['0', '1', 'K'] if mode == 'tokens' else ['0', '1', 'M', 'a']
        inputs = [rf'$\mathbf{{e}}_{{{index}}}$' for index in indices]
        labels = ['Question\n', r'$x_1$', r'$x_K$'] if mode == 'tokens' else ['Question\n', r'$\mathbf{z}_1$', r'$\mathbf{z}_M$', '']
        grid = {}
        for column, (x, value, index) in enumerate(zip(positions, inputs, indices)):
            for row, y in enumerate([0, 3, 5]):
                fill = '0.93' if row == 0 else '#efedf8'
                text = ax.text(x, y, value if row == 0 else rf'$\mathbf{{h}}_{{{index}}}^\ell$', ha='center', va='center', zorder=4)
                texts.append((ax, text, fill, 'state'))
                grid[column, row] = text
        answer = ax.text(positions[-1], 7.2, r'$y_1$', ha='center', va='center', zorder=4)
        texts.append((ax, answer, '#e5eff9', 'payload'))
        payloads = [r'$x_1$', r'$x_K$'] if mode == 'tokens' else [r'$\mathbf{z}_1$', r'$\mathbf{z}_M$']
        for k, (x, label) in enumerate(zip(positions[:2], payloads)):
            text = ax.text(x + 1.25, 1.5, label, ha='center', va='center', zorder=4)
            texts.append((ax, text, '#e5eff9' if mode == 'tokens' else '#fff0d9', 'payload'))
            glyphs.append((ax, text, mode, k))
        ax.set(xlim=(-1.2, 9.8), ylim=(-2.2, 8.1), aspect='equal', xlabel=r'Position $t$', ylabel=r'Layer $\ell$')
        ax.set_xticks(positions, [rf'${index}$' + '\n' + label for index, label in zip(indices, labels)])
        ax.set_yticks([0, 3, 5], ['0', r'$L-1$', r'$L$'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_bounds(0, 5)
        ax.spines['bottom'].set_bounds(positions[0], positions[-1])
        ax.set_title(title, fontweight='bold')
        panels.append((mode, positions, grid, answer))
    # Fit circles tightly around default-size labels, with uniform sizes in
    # each group. Shorter payload labels keep those nodes smaller than states.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    radius_px = {group: max(hypot(text.get_window_extent(renderer).width, text.get_window_extent(renderer).height) / 2 for _, text, _, kind in texts if kind == group) for group in ['state', 'payload']}
    circles = {}
    for ax, text, fill, group in texts:
        center = ax.transData.transform(text.get_position())
        radius = ax.transData.inverted().transform(center + [radius_px[group], 0])[0] - text.get_position()[0]
        circle = Circle(text.get_position(), radius, facecolor=fill, edgecolor='0.2', zorder=3)
        ax.add_patch(circle)
        circles[text] = circle
    # The vector spans the payload node's diameter, right of the node: borderless cells inside one outline.
    for ax, text, mode, k in glyphs:
        (cx, cy), radius = text.get_position(), circles[text].radius
        cells = len(GLYPH_VALUES[k])
        cell = 2 * radius / cells
        left, width = cx + radius + cell / 2, 1.5 * cell
        for i in range(cells):
            fill = (COLORS['token'] if i == GLYPH_HOT[k] else 'white') if mode == 'tokens' else to_rgba(COLORS['feedback'], GLYPH_VALUES[k][i])
            ax.add_patch(Rectangle((left, cy + radius - (i + 1) * cell), width, cell, facecolor=fill, edgecolor='none', zorder=3))
        for i in range(1, cells):
            ax.plot([left, left + width], [cy + radius - i * cell] * 2, color='0.2', linewidth=plt.rcParams['patch.linewidth'] / 2, solid_capstyle='butt', zorder=3)
        ax.add_patch(Rectangle((left, cy - radius), width, 2 * radius, fill=False, edgecolor='0.2', zorder=3))
    for ax, (mode, positions, grid, answer) in zip(axes, panels):
        for column, x in enumerate(positions):
            color = COLORS['path'] if (mode == 'tokens' and column == 0) or (mode == 'latent' and column < 3) else COLORS['attention']
            arrow(ax, circles[grid[column, 0]], circles[grid[column, 1]], color)
            ax.text(x, 1.5, r'$\vdots$', ha='center', va='center', zorder=5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'boxstyle': 'square,pad=0'})
            top_color = COLORS['attention'] if mode == 'latent' and column == 2 else COLORS['path'] if mode == 'latent' and column == 3 else color
            arrow(ax, circles[grid[column, 1]], circles[grid[column, 2]], top_color)
        # Show only neighboring cached-attention connections to keep the grid clear.
        for column in range(1, len(positions)):
            color = COLORS['path'] if mode == 'latent' and column == 3 else COLORS['attention']
            arrow(ax, circles[grid[column - 1, 1]], circles[grid[column, 2]], color)
        for column in range(2):
            feedback(ax, circles[grid[column, 2]], circles[grid[column + 1, 0]], COLORS['token'] if mode == 'tokens' else COLORS['feedback'])
        for y in [0, 3, 5]: ax.text((positions[1] + positions[2]) / 2, y, r'$\cdots$', ha='center', va='center', zorder=5)
        arrow(ax, circles[grid[len(positions) - 1, 2]], circles[answer], COLORS['token'])
    fig.savefig(out / 'codi_flow.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    reproduce(pathlib.Path(os.environ.get("FIG_DIR", ROOT / "figs")))
