from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

Node = Dict[str, Any]

def _layout(node: Node, x=0.0, y=0.0, x_spacing=1.5, y_spacing=1.8) -> Tuple[List[Tuple[Node, float, float]], float]:
    """Compute (node, x, y) positions and return them with the total width."""
    if node.get("leaf", False):
        return [ (node, x, y) ], 1.0
    left_layout, left_w = _layout(node["left"], x, y - y_spacing, x_spacing, y_spacing)
    right_layout, right_w = _layout(node["right"], x + left_w * x_spacing, y - y_spacing, x_spacing, y_spacing)
    # Center this node above its children
    node_x = (left_layout[0][1] + right_layout[-1][1]) / 2.0
    return [ (node, node_x, y) ] + left_layout + right_layout, left_w + right_w

def draw_tree(node: Node, feature_names=None, class_names=None, figsize=(14, 8), filename=None):
    if feature_names is None:
        feature_names = [f"A{i}" for i in range(7)]
    if class_names is None:
        class_names = {1: "Room 1", 2: "Room 2", 3: "Room 3", 4: "Room 4"}

    positions, _ = _layout(node, x=0.0, y=0.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Draw edges first
    pos_map = {id(n): (xx, yy) for (n, xx, yy) in positions}
    for (n, x, y) in positions:
        if not n.get("leaf", False):
            for child_key, label in [("left", "≤"), ("right", ">")]:
                c = n[child_key]
                cx, cy = pos_map[id(c)]
                ax.plot([x, cx], [y, cy], linewidth=1.0)
                midx, midy = (x+cx)/2, (y+cy)/2
                txt = f"{label} {n['threshold']:.2f}" if label == "≤" else None
                if txt:
                    ax.text(midx, midy+0.15, txt, fontsize=8, ha='center', va='bottom')

    # Draw nodes
    for (n, x, y) in positions:
        if n.get("leaf", False):
            txt = f"Leaf\nclass={n['prediction']}"
        else:
            attr = n['attr']
            txt = f"{feature_names[attr]}"
        bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
        ax.text(x, y, txt, ha='center', va='center', fontsize=9, bbox=bbox)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
    return fig, ax
