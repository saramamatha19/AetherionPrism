"""Draw the two result charts shown in README.md (a light and a dark version of each).

The numbers are copied from the results files (sources in the comments). If a result changes,
change the number here and run again:

    uv run --with matplotlib python docs/make_charts.py

Writes docs/images/<chart>_light.svg and docs/images/<chart>_dark.svg.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
matplotlib.rcParams["svg.hashsalt"] = "prism"  # same ids every run (no noise in git diffs)
import matplotlib.pyplot as plt

OUT = Path("docs/images")

# Exactly right on the test set: (name, real-world %, hard cases %)
# Source: results/laya_comparison.md (model B, gpt-4o v4, full system rows)
SYSTEMS = [
    ("Small model alone", 84.0, 40.4),
    ("LLM alone (gpt-4o)", 74.6, 71.4),
    ("Prism: model + LLM", 84.2, 52.1),
]

# Every gate setting: (gate, model answers %, all exact %, real-world exact %, $ per 1M)
# Source: results/cost_accuracy_curve.md
CURVE = [
    (0.00, 91, 66.0, 83.6, 162),
    (0.05, 85, 68.0, 84.4, 265),
    (0.10, 79, 69.2, 84.4, 362),
    (0.13, 76, 70.0, 84.2, 414),
    (0.15, 75, 70.3, 84.0, 443),
    (0.20, 70, 71.8, 84.2, 517),
    (0.25, 66, 72.8, 84.4, 585),
    (0.30, 62, 73.8, 83.8, 664),
    (0.40, 55, 74.3, 82.0, 785),
    (0.50, 48, 74.8, 81.2, 906),
    (0.60, 42, 75.1, 79.8, 1007),
    (0.80, 26, 74.6, 76.6, 1289),
    (0.95, 2, 74.5, 74.8, 1699),
]
TODAY_GATE = 0.13
LLM_ONLY = (1731, 74.4, 74.6)  # $ per 1M, all exact %, real-world exact %

# Colours for each theme: text, faint text, grid lines, series 1 (real-world), series 2 (hard / all)
THEMES = {
    "light": {"ink": "#15202b", "muted": "#5b6672", "grid": "#d6dce3", "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"ink": "#e5ebef", "muted": "#9aa6b2", "grid": "#33414c", "s1": "#3987e5", "s2": "#d95926"},
}


def style_axes(ax, colours):
    """Quiet axes: no box, faint grid, theme-coloured text."""
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(colours["grid"])
    ax.tick_params(colors=colours["muted"], labelsize=10, length=0)
    ax.grid(axis="y", color=colours["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def systems_chart(theme: str) -> None:
    """Grouped bars: real-world vs hard questions, for the model, the LLM and both together."""
    colours = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.36
    for i in range(len(SYSTEMS)):
        name, real_world, hard = SYSTEMS[i]
        bars = [
            (i - width / 2, real_world, colours["s1"], "Real-world questions (500)"),
            (i + width / 2, hard, colours["s2"], "Hard trick questions (560)"),
        ]
        for x, value, colour, label in bars:
            if i > 0:
                label = None  # one legend entry per series
            ax.bar(x, value, width - 0.03, color=colour, label=label)
            ax.text(x, value + 1.5, f"{value:.1f}%", ha="center", color=colours["ink"], fontsize=10)

    names = []
    for name, _, _ in SYSTEMS:
        names.append(name)
    ax.set_xticks(range(len(SYSTEMS)))
    ax.set_xticklabels(names, color=colours["ink"], fontsize=11)
    ax.set_ylim(0, 112)  # room above the bars for the legend
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    style_axes(ax, colours)
    ax.set_title(
        "Exactly right on the test set: the model wins on normal questions,\n"
        "the LLM on hard ones, and Prism keeps the best of both",
        color=colours["ink"], fontsize=12, loc="left",
    )
    ax.legend(frameon=False, labelcolor=colours["ink"], fontsize=10, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / f"systems_{theme}.svg", transparent=True, metadata={"Date": None})
    plt.close(fig)


def curve_chart(theme: str) -> None:
    """Cost vs accuracy for every gate, with today's gate and 'LLM every time' marked."""
    colours = THEMES[theme]
    costs = []
    real_world = []
    all_exact = []
    for gate, _, all_value, real_value, cost in CURVE:
        costs.append(cost)
        all_exact.append(all_value)
        real_world.append(real_value)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(costs, real_world, color=colours["s1"], linewidth=2, label="Real-world questions")
    ax.plot(costs, all_exact, color=colours["s2"], linewidth=2, label="All questions")

    for gate, share, all_value, real_value, cost in CURVE:
        if gate == TODAY_GATE:
            for value in [real_value, all_value]:
                ax.scatter(cost, value, s=110, facecolors="none", edgecolors=colours["ink"], linewidths=2, zorder=3)
            ax.annotate(
                f"today: gate {gate}, {share}% answered by the model, ${cost} per 1M",
                (cost, real_value), xytext=(cost + 120, real_value + 3.5),
                color=colours["ink"], fontsize=10,
                arrowprops={"arrowstyle": "-", "color": colours["muted"], "linewidth": 1},
            )

    llm_cost, llm_all, llm_real = LLM_ONLY
    ax.scatter([llm_cost], [llm_real], marker="s", s=60, color=colours["s1"], zorder=3)
    ax.scatter([llm_cost], [llm_all], marker="s", s=60, color=colours["s2"], zorder=3)
    ax.annotate("LLM every time, $1,731", (llm_cost, llm_all), xytext=(llm_cost - 520, llm_all - 4.5),
                color=colours["ink"], fontsize=10,
                arrowprops={"arrowstyle": "-", "color": colours["muted"], "linewidth": 1})

    ax.set_xlim(100, 1850)
    ax.set_ylim(60, 92)
    ax.set_xticks([300, 600, 900, 1200, 1500, 1800])
    ax.set_xticklabels(["$300", "$600", "$900", "$1,200", "$1,500", "$1,800"])
    ax.set_yticks([60, 65, 70, 75, 80, 85, 90])
    ax.set_yticklabels(["60%", "65%", "70%", "75%", "80%", "85%", "90%"])
    ax.set_xlabel("Cost per 1 million questions (more questions sent to the LLM →)",
                  color=colours["muted"], fontsize=10)
    style_axes(ax, colours)
    ax.set_title("The gate is a dial between cost and accuracy (exactly right, test set)",
                 color=colours["ink"], fontsize=12, loc="left")
    ax.legend(frameon=False, labelcolor=colours["ink"], fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"cost_accuracy_{theme}.svg", transparent=True, metadata={"Date": None})
    plt.close(fig)


PRISM_SOURCES = ["kb", "jira", "confluence", "slack", "notion", "gmail", "web"]
PRISM_THEMES = {
    "light": {"ink": "#15202b", "muted": "#5b6672", "fill": "#ffffff",
              "beams": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]},
    "dark": {"ink": "#e5ebef", "muted": "#9aa6b2", "fill": "#161b22",
             "beams": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2f9e2f", "#9085e9"]},
}


def prism_picture(theme: str) -> None:
    """The README's title picture: a question goes into a prism and splits into the 7 sources."""
    colours = PRISM_THEMES[theme]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 210" width="340" height="210" '
        'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">',
        f'<text x="8" y="94" font-size="12" fill="{colours["muted"]}">a question</text>',
        f'<line x1="8" y1="105" x2="97" y2="105" stroke="{colours["muted"]}" stroke-width="2.5" '
        'stroke-linecap="round"/>',
        f'<polygon points="130,25 75,165 185,165" fill="{colours["fill"]}" stroke="{colours["ink"]}" '
        'stroke-width="2" stroke-linejoin="round"/>',
        f'<line x1="97" y1="105" x2="162" y2="105" stroke="{colours["muted"]}" stroke-width="1.5" '
        'stroke-dasharray="3 3"/>',
    ]
    for i in range(len(PRISM_SOURCES)):
        y = 20 + i * 28
        parts.append(
            f'<line x1="162" y1="105" x2="246" y2="{y}" stroke="{colours["beams"][i]}" '
            'stroke-width="2.6" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="254" y="{y + 4}" font-size="13" fill="{colours["ink"]}">{PRISM_SOURCES[i]}</text>'
        )
    parts.append("</svg>")
    (OUT / f"prism_{theme}.svg").write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        systems_chart(theme)
        curve_chart(theme)
        prism_picture(theme)
        print(f"wrote {theme} charts to {OUT}/")
