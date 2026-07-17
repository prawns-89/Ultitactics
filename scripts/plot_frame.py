"""Draw a synthetic vertical-stack frame, so you can see what the model sees.

    python scripts/plot_frame.py            # random frame -> frame.png
    python scripts/plot_frame.py --seed 7 --out myframe.png

Each throw is annotated with its ground-truth completion probability, and the
best read is highlighted. Useful for sanity-checking that the simulator produces
formations that look like actual Ultimate, and for building intuition about what
the completion rule rewards before you trust any model output.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ultitactics import field, simulate
from ultitactics.formation import best_receiver, completion_probs


def plot_frame(frame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    # -- Field: playing proper + both endzones. ----------------------------
    ax.add_patch(
        plt.Rectangle(
            (-field.HALF_LENGTH, -field.HALF_WIDTH),
            field.TOTAL_LENGTH,
            field.WIDTH,
            facecolor="#eef5ee",
            edgecolor="#557755",
        )
    )
    for sign in (-1, 1):
        ax.axvline(sign * field.GOAL_LINE_X, color="#557755", lw=1.2)

    probs = completion_probs(frame)
    best = best_receiver(frame)
    thrower = frame.thrower_pos

    # -- Throwing lanes, shaded by completion probability. ------------------
    for cand, p in zip(field.IDX_RECEIVER_CANDIDATES, probs):
        target = frame.pos[cand]
        is_best = cand == best
        ax.plot(
            [thrower[0], target[0]],
            [thrower[1], target[1]],
            color="#d1495b" if is_best else "#888888",
            lw=2.5 if is_best else 1.0,
            alpha=0.9 if is_best else 0.35,
            zorder=2,
        )
        mid = (thrower + target) / 2
        ax.text(
            mid[0], mid[1] + 0.6, f"{p:.2f}",
            fontsize=8, ha="center",
            color="#d1495b" if is_best else "#666666",
            fontweight="bold" if is_best else "normal",
            zorder=5,
        )

    # -- Players. ----------------------------------------------------------
    for i in range(field.N_PLAYERS):
        p = frame.pos[i]
        offence = field.IS_OFFENCE[i]
        ax.scatter(
            p[0], p[1],
            s=170, zorder=4,
            c="#2e6f9e" if offence else "#c1666b",
            marker="o" if offence else "s",
            edgecolors="black", linewidths=0.8,
        )
        rank = field.STACK_RANK[i]
        label = {field.IDX_THROWER: "T", field.IDX_RESET: "R"}.get(
            i, str(rank) if rank else ""
        )
        if i == field.IDX_MARK:
            label = "M"
        if label:
            ax.text(p[0], p[1], label, fontsize=7, ha="center", va="center",
                    color="white", fontweight="bold", zorder=5)
        # Velocity arrow for anyone actually moving.
        v = frame.vel[i]
        if np.linalg.norm(v) > 0.3:
            ax.arrow(p[0], p[1], v[0] * 0.6, v[1] * 0.6,
                     head_width=0.7, color="#2e6f9e", zorder=3, alpha=0.8)

    # -- Annotations. ------------------------------------------------------
    ax.arrow(
        thrower[0], -field.HALF_WIDTH + 1.5, frame.attack_dir * 6, 0,
        head_width=1.0, color="#333333",
    )
    ax.text(
        thrower[0] + frame.attack_dir * 3, -field.HALF_WIDTH + 2.6,
        "attacking", fontsize=8, ha="center",
    )
    open_side = "+y" if frame.force_open_y > 0 else "-y"
    ax.set_title(
        f"Vertical stack  |  stall {frame.stall}  |  open side {open_side}  |  "
        f"wind ({frame.wind[0]:.1f}, {frame.wind[1]:.1f}) m/s\n"
        f"circles = offence, squares = defence, numbers = stack rank; "
        f"red lane = best read",
        fontsize=10,
    )
    ax.set_xlim(-field.HALF_LENGTH - 2, field.HALF_LENGTH + 2)
    ax.set_ylim(-field.HALF_WIDTH - 2, field.HALF_WIDTH + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m) - goal to goal")
    ax.set_ylabel("y (m) - sideline to sideline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="frame.png")
    args = ap.parse_args()

    frame = simulate.sample_frame(np.random.default_rng(args.seed))
    probs = completion_probs(frame)
    print("candidate completion probabilities (0=reset, 1..5=cutter rank):")
    for cand, p in zip(field.IDX_RECEIVER_CANDIDATES, probs):
        rank = field.STACK_RANK[cand]
        name = "reset" if cand == field.IDX_RESET else f"cutter rank {rank}"
        print(f"  {name:16} {p:.3f}")
    plot_frame(frame, args.out)


if __name__ == "__main__":
    main()
