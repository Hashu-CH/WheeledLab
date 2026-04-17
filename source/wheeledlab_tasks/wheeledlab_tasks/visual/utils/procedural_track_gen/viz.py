"""
Visualisation helpers for curriculum track generation.
"""

import matplotlib.pyplot as plt
from .curriculum import CurriculumConfig, generate_track


def plot_curriculum_progression(figsize=(14, 6), show=True):
    """
    Sample curriculum progression: show tracks at increasing difficulties.
    One row, multiple difficulty columns.
    """
    difficulties = [0.0, 0.2, 0.4, 0.65, 0.8, 1.0]
    n_cols = len(difficulties)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    for col, d in enumerate(difficulties):
        cfg = CurriculumConfig(difficulty=d)
        grid, meta = generate_track(config=cfg)
        ax = axes[col]
        ax.imshow(grid, cmap="gray", interpolation="nearest")

        phase = meta["phase"]
        if phase == "chain":
            feats = [f["feature"] for f in meta["features"]]
            label = ", ".join(feats[:2]) + ("…" if len(feats) > 2 else "")
        else:
            label = f"loop"

        ax.set_title(f"d={d:.2f}\n{phase}\n{label}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Red border at phase transition
        if d == 0.65:
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

    plt.suptitle("Curriculum Progression (difficulty 0→1, phase change at d=0.65)", fontsize=11)
    plt.tight_layout()
    if show:
        plt.show()
    print("Saved curriculum_progression.png")
