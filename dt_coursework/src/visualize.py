from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional

Node = Dict[str, Any]

def _layout_tidy(
    node: Node,
    depth: int = 0,
    _leaf_counter: List[int] = None,
) -> Tuple[List[Tuple[Node, float, int]], float, float, float]:
    '''
    parameters:
        node (Node): Current or root node of the decision tree.
        depth (int): Current depth level in the tree. Default is 0.
        _leaf_counter (List[int]): Counter to assign x-coordinates to leaves in order.

    functionality:
        Recursively computes a clean, non-overlapping layout of the tree by assigning
        x/y positions to each node based on in-order traversal.

    return:
        Tuple containing:
            - positions (List[Tuple[Node, float, int]]): Node with its x-position and depth.
            - xmin (float): Minimum x-coordinate among all leaves.
            - xmax (float): Maximum x-coordinate among all leaves.
            - xcenter (float): Center x-coordinate of the current subtree.
    '''
    if _leaf_counter is None:
        _leaf_counter = [0]

    # Base case: if leaf node, assign x position and increment counter
    if node.get("leaf", False):
        x = float(_leaf_counter[0])
        _leaf_counter[0] += 1
        return [(node, x, depth)], x, x, x

    # Recursively compute layout for left and right subtrees
    left_pos, lmin, lmax, lcen = _layout_tidy(node["left"], depth + 1, _leaf_counter)
    right_pos, rmin, rmax, rcen = _layout_tidy(node["right"], depth + 1, _leaf_counter)

    # Compute x position as midpoint of left and right children
    xcenter = (lcen + rcen) / 2.0
    here = [(node, xcenter, depth)]

    # Combine current node and children positions
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
    '''
    parameters:
        node (Node): Root node of the decision tree to visualize.
        feature_names (List[str], optional): List of feature names for internal nodes.
        class_names (Dict[int, str], optional): Mapping from class indices to labels.
        figsize (Tuple[float, float], optional): Figure size for the plot.
        filename (str, optional): Path to save the visualization.
        x_spacing (float): Horizontal spacing multiplier.
        y_spacing (float): Vertical spacing multiplier.
        leaf_gap_units (float): Extra horizontal spacing between leaves.
        annotate_thresholds (bool): If True, show threshold values on edges.

    functionality:
        Draws a complete visual representation of a decision tree using Matplotlib,
        including nodes, edges, and threshold annotations. Uses tidy layout for positioning.

    return:
        Tuple:
            - fig (matplotlib.figure.Figure): Created figure object.
            - ax (matplotlib.axes.Axes): Axis object containing the tree plot.
    '''
    if feature_names is None:
        feature_names = [f"A{i}" for i in range(7)]
    if class_names is None:
        class_names = {1: "Room 1", 2: "Room 2", 3: "Room 3", 4: "Room 4"}

    # Get tidy layout positions for each node
    positions, xmin, xmax, _ = _layout_tidy(node, depth=0)
    n_leaves = int(xmax - xmin + 1)

    # Auto-adjust figure size if not provided
    if figsize is None:
        max_depth = max(d for _, _, d in positions)
        w = max(14.0, n_leaves * 0.7 * (1.0 + leaf_gap_units))
        h = max(7.0, (max_depth + 1) * 1.2)
        figsize = (w, h)

    # Convert leaf index positions to actual plot coordinates
    unit = x_spacing * (1.0 + leaf_gap_units)
    coords = {}
    for (n, x_idx, depth) in positions:
        x = (x_idx - xmin) * unit
        y = -depth * y_spacing
        coords[id(n)] = (x, y)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Draw connecting edges
    for (n, _, _) in positions:
        if not n.get("leaf", False):
            x, y = coords[id(n)]
            for child_key, side_label in [("left", "≤"), ("right", ">")]:
                c = n[child_key]
                cx, cy = coords[id(c)]
                ax.plot([x, cx], [y, cy], linewidth=1.0)

                # Optionally annotate thresholds on edges
                if annotate_thresholds:
                    midx, midy = (x + cx) / 2.0, (y + cy) / 2.0
                    if side_label == "≤":
                        txt = f"≤ {n['threshold']:.2f}"
                    else:
                        txt = f">{n['threshold']:.2f}"
                    ax.text(midx, midy + 0.12, txt, fontsize=8, ha="center", va="bottom")

    # Draw nodes (boxes) after edges
    for (n, _, _) in positions:
        x, y = coords[id(n)]
        if n.get("leaf", False):
            txt = f"Room\n{n['prediction']}"
        else:
            attr = n["attr"]
            txt = f"{feature_names[attr]}"
        bbox = dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1)
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, bbox=bbox)

    # Save figure if filename provided
    if filename:
        fig.savefig(filename, dpi=75, bbox_inches="tight")

    return fig, ax
