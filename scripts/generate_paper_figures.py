"""Generate standalone publication-quality PDF and PNG figures for the ReviTLS IEEE paper."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generate_all_figures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })

    # 1. Grouped Bar Chart: 7 Systems across 4 Core Metrics
    systems = [
        "LLM-TLS\n(ACL'24)",
        "CHRONOS\n(NAACL'25)",
        "ReviTLS\n(Llama-8B)",
        "ReviTLS\n(Gemma-12B)",
        "ReviTLS\n(Qwen-14B)",
        "ReviTLS\n(Qwen-7B)",
        "ReviTLS-Hybrid\n(Ours: 7B+14B)",
    ]
    date_f1 = [0.6000, 0.1905, 0.2222, 0.1176, 0.3158, 0.4375, 0.5714]
    concat_r1 = [0.4098, 0.2315, 0.3116, 0.3874, 0.4427, 0.5397, 0.5594]
    rra = [0.0000, 0.2500, 0.5000, 0.2500, 0.7500, 0.7500, 0.7500]
    leak_free = [0.0000, 0.7500, 0.7500, 0.5000, 1.0000, 1.0000, 0.7500]

    x = np.arange(len(systems))
    w = 0.20

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.bar(x - 1.5 * w, date_f1, w, label="Date-F1 (↑)", color="#2b5c8f", edgecolor="black", linewidth=0.6)
    ax.bar(x - 0.5 * w, concat_r1, w, label="Concat-R1 (↑)", color="#2ca02c", edgecolor="black", linewidth=0.6)
    ax.bar(x + 0.5 * w, rra, w, label="Revision Accuracy RRA (↑)", color="#e67e22", edgecolor="black", linewidth=0.6)
    ax.bar(x + 1.5 * w, leak_free, w, label="Leak-Free Rate 1-SFLR (↑)", color="#7f3c8d", edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Normalized Metric Score [0, 1]")
    ax.set_title("Multi-Model & Baseline Performance Profile on T17/bpoil + Revision Stress Stream")
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_multi_model_metrics.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_multi_model_metrics.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Pareto Frontier: Latency vs Date-F1
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    points = [
        ("Llama-3.1-8B", 444.07, 0.2222, "#e67e22", "o"),
        ("Gemma-3-12B", 1067.72, 0.1176, "#c0392b", "^"),
        ("Qwen2.5-14B", 706.32, 0.3158, "#8e44ad", "s"),
        ("Qwen2.5-7B", 619.13, 0.4375, "#2980b9", "D"),
        ("ReviTLS-Hybrid (Ours)", 488.50, 0.5714, "#27ae60", "*"),
    ]
    for label, t_sec, df1, color, marker in points:
        size = 180 if "Hybrid" in label else 90
        ax.scatter(t_sec, df1, s=size, color=color, marker=marker, edgecolors="black", zorder=4, label=label)
        ax.annotate(f" {label}\n ({t_sec:.0f}s, {df1:.3f})", (t_sec, df1), fontsize=8, xytext=(5, -4), textcoords="offset points")

    ax.annotate("", xy=(488.50, 0.5714), xytext=(619.13, 0.4375),
                arrowprops=dict(arrowstyle="->", linestyle="--", color="#27ae60", lw=1.5))
    ax.annotate("", xy=(488.50, 0.5714), xytext=(706.32, 0.3158),
                arrowprops=dict(arrowstyle="->", linestyle="--", color="#27ae60", lw=1.5))

    ax.set_xlabel("End-to-End Inference Latency on RTX 5080 (Seconds) ↓")
    ax.set_ylabel("Temporal Accuracy (Date-F1) ↑")
    ax.set_xlim(390, 1180)
    ax.set_ylim(0.05, 0.65)
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_pareto_latency.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig4_pareto_latency.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Step-by-Step Component Ablation Trajectory
    stages = ["A0: LLM-TLS", "A1: v1 (90d)", "A2: +FactPurge", "A3: +CondGate", "A4: +CrossEnc", "A5: Hybrid"]
    s_df1 = [0.6000, 0.4375, 0.4375, 0.5263, 0.5500, 0.5714]
    s_cr1 = [0.4098, 0.5210, 0.5397, 0.5482, 0.5531, 0.5594]
    s_rra = [0.0000, 0.2500, 0.7500, 0.7500, 0.7500, 0.7500]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(stages, s_df1, marker="o", lw=2, color="#2b5c8f", label="Date-F1 (↑)")
    ax.plot(stages, s_cr1, marker="s", lw=2, color="#2ca02c", label="Concat-R1 (↑)")
    ax.plot(stages, s_rra, marker="^", lw=2, color="#e67e22", label="Revision Accuracy RRA (↑)")
    ax.set_ylim(-0.05, 0.85)
    ax.set_ylabel("Metric Score")
    ax.set_title("Step-by-Step Architectural Ablation Trajectory")
    ax.grid(True, linestyle="--", alpha=0.45)
    plt.xticks(rotation=18)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_stepwise_ablation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig5_stepwise_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved publication figures to {out_dir.resolve()}")


if __name__ == "__main__":
    generate_all_figures(Path("paper/figures"))
