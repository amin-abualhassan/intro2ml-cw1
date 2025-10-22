from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional

Node = Dict[str, Any]

def _layout_tidy(
    node: Node,
    depth: int = 0,
    _leaf_counter: List[int] = None,
) -> Tuple[List[Tuple[Node, float, int]], float, float, float]:
    """
    Tidy tree layout:
    - Give every leaf a unique x index by in-order assignment.
    - Set internal node x as the midpoint of its children's x.
    Returns:
      positions: list of (node, x_index, depth)
      xmin, xmax: leaf-index bounds for this subtree
      xcenter: center (in leaf-index units) for this subtree
    """
    if _leaf_counter is None:
        _leaf_counter = [0]

    if node.get("leaf", False):
        x = float(_leaf_counter[0])
        _leaf_counter[0] += 1
        return [(node, x, depth)], x, x, x

    # recurse (left, right)
    left_pos, lmin, lmax, lcen = _layout_tidy(node["left"], depth + 1, _leaf_counter)
    right_pos, rmin, rmax, rcen = _layout_tidy(node["right"], depth + 1, _leaf_counter)

    xcenter = (lcen + rcen) / 2.0
    here = [(node, xcenter, depth)]
    positions = left_pos + right_pos + here
    return positions, min(lmin, rmin), max(lmax, rmax), xcenter


def draw_tree(
    node: Node,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[Dict[int, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    filename: Optional[str] = None,
    x_spacing: float = 1.1,
    y_spacing: float = 1.1,
    leaf_gap_units: float = 2.0,
    annotate_thresholds: bool = True,
):
    """
    Render a tidy binary tree with Matplotlib.

    Parameters
    ----------
    x_spacing, y_spacing : float
        Scale factors for horizontal / vertical space between nodes.
    annotate_thresholds : bool
        If True, draw '≤ threshold' on left edges and '>' on right edges.
    """
    if feature_names is None:
        feature_names = [f"A{i}" for i in range(7)]
    if class_names is None:
        class_names = {1: "Room 1", 2: "Room 2", 3: "Room 3", 4: "Room 4"}

    positions, xmin, xmax, _ = _layout_tidy(node, depth=0)
    n_leaves = int(xmax - xmin + 1)

    # Auto figure size if not provided: widen with number of leaves
    if figsize is None:
        max_depth = max(d for _, _, d in positions)
        w = max(14.0, n_leaves * 0.7 * (1.0 + leaf_gap_units))  # include leaf gaps
        h = max(7.0,  (max_depth + 1) * 1.2)
        figsize = (w, h)

    # Convert from leaf-index space to plot coords
    unit = x_spacing * (1.0 + leaf_gap_units)  # <<< each adjacent leaf is farther apart
    coords = {}
    for (n, x_idx, depth) in positions:
        x = (x_idx - xmin) * unit
        y = -depth * y_spacing
        coords[id(n)] = (x, y)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Draw edges first
    for (n, _, _) in positions:
        if not n.get("leaf", False):
            x, y = coords[id(n)]
            for child_key, side_label in [("left", "≤"), ("right", ">")]:
                c = n[child_key]
                cx, cy = coords[id(c)]
                ax.plot([x, cx], [y, cy], linewidth=1.0)
                if annotate_thresholds:
                    midx, midy = (x + cx) / 2.0, (y + cy) / 2.0
                    if side_label == "≤":
                        txt = f"≤ {n['threshold']:.2f}"
                    else:
                        txt = f">{n['threshold']:.2f}"
                    ax.text(midx, midy + 0.12, txt, fontsize=8, ha="center", va="bottom")

    # Draw nodes (after edges so boxes are on top)
    for (n, _, _) in positions:
        x, y = coords[id(n)]
        if n.get("leaf", False):
            txt = f"Room\n{n['prediction']}"
        else:
            attr = n["attr"]
            txt = f"{feature_names[attr]}"
        bbox = dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1)
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, bbox=bbox)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig, ax

