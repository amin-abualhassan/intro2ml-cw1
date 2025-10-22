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
    - Input:
        node (Node): The root node (or current subtree node) of the decision tree.
        depth (int): Current depth level in the tree. Default is 0.
        _leaf_counter (List[int]): Internal counter used to assign x-coordinates to leaves in-order.
    
    - Process:
        Recursively traverses the decision tree to compute a "tidy" layout.
        Each leaf is assigned a unique x-coordinate based on its in-order position.
        For internal nodes, the x-coordinate is the midpoint between its left and right children.
    
    - Return:
        Tuple:
            - positions (List[Tuple[Node, float, int]]): List of nodes with their x-coordinate and depth.
            - xmin (float): Minimum x-coordinate among leaves.
            - xmax (float): Maximum x-coordinate among leaves.
            - xcenter (float): Center x-coordinate of the current subtree.
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
    - Input:
        node (Node): Root node of the trained decision tree.
        feature_names (List[str], optional): Names of the features for labeling internal nodes.
        class_names (Dict[int, str], optional): Mapping from class indices to human-readable labels.
        figsize (Tuple[float, float], optional): Custom figure size for the visualization.
        filename (str, optional): Path to save the resulting tree plot.
        x_spacing (float): Horizontal spacing multiplier between nodes.
        y_spacing (float): Vertical spacing multiplier between levels.
        leaf_gap_units (float): Extra horizontal spacing between leaves.
        annotate_thresholds (bool): Whether to show threshold values (≤ / >) on connecting edges.
    
    - Process:
        Uses a tidy layout from '_layout_tidy()' to compute node coordinates.
        Draws the decision tree using Matplotlib:
            • Connects parent and child nodes with edges.
            • Labels edges with threshold conditions if enabled.
            • Displays nodes as boxes showing either feature names (internal nodes)
              or class predictions (leaf nodes).
        Optionally saves the figure to a file if 'filename' is provided.
    
    - Return:
        Tuple:
            - fig (matplotlib.figure.Figure): The created figure object.
            - ax (matplotlib.axes.Axes): The axis object containing the visualization.
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

