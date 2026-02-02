import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _apply_publication_style(fig):
    """
    Make plots paper-ready:
    - white background
    - black square border (box) around each subplot axes
    - no grid
    """
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        showgrid=False,
        zeroline=False,
    )
    return fig


def plot_accuracy_loss_across_seeds(dict_metrics_seed, cf):
    """
    Creates a 2x2 Plotly figure:
      Top row:    Accuracy (Train, Test)
      Bottom row: Loss_total (Train, Test)

    Each subplot shows mean ± 1 std across seeds.

    dict_metrics_seed[seed]["train"]["loss_train"] -> list of [loss_total, loss_ce, loss_bce] per epoch
    dict_metrics_seed[seed]["test"]["loss_test"]   -> list of [loss_total, loss_ce, loss_bce] per epoch
    dict_metrics_seed[seed]["train"]["acc_train"]  -> list of accuracy per epoch
    dict_metrics_seed[seed]["test"]["acc_test"]    -> list of accuracy per epoch
    """
    seeds = sorted(dict_metrics_seed.keys())

    def get_loss_total(seed, split):
        key = "loss_train" if split == "train" else "loss_test"
        lst = dict_metrics_seed[seed][split][key]
        return np.array([x[0] for x in lst], dtype=float)

    def get_acc(seed, split):
        key = "acc_train" if split == "train" else "acc_test"
        return np.array(dict_metrics_seed[seed][split][key], dtype=float)

    # stack arrays: (S, E)
    loss_train = np.stack([get_loss_total(s, "train") for s in seeds], axis=0)
    loss_test = np.stack([get_loss_total(s, "test") for s in seeds], axis=0)
    acc_train = np.stack([get_acc(s, "train") for s in seeds], axis=0)
    acc_test = np.stack([get_acc(s, "test") for s in seeds], axis=0)

    E = loss_train.shape[1]
    epochs = np.arange(1, E + 1)

    def mean_std(x):
        return x.mean(axis=0), x.std(axis=0)

    m_lt, s_lt = mean_std(loss_train)
    m_lte, s_lte = mean_std(loss_test)
    m_at, s_at = mean_std(acc_train)
    m_ate, s_ate = mean_std(acc_test)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Train Accuracy",
            "Test Accuracy",
            "Train Loss_total",
            "Test Loss_total",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
    )

    def add_mean_std(row, col, x, mean, std, name, showlegend):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines+markers",
                name=name,
                showlegend=showlegend,
                customdata=np.stack([std], axis=-1),
                hovertemplate=(
                    "epoch=%{x}<br>"
                    "mean=%{y:.6f}<br>"
                    "std=%{customdata[0]:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        upper = mean + std
        lower = mean - std
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                line=dict(width=0),
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    add_mean_std(1, 1, epochs, m_at, s_at, "Train Accuracy (mean)", True)
    add_mean_std(1, 2, epochs, m_ate, s_ate, "Test Accuracy (mean)", True)
    add_mean_std(2, 1, epochs, m_lt, s_lt, "Train Loss_total (mean)", True)
    add_mean_std(2, 2, epochs, m_lte, s_lte, "Test Loss_total (mean)", True)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)

    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="Loss_total", row=2, col=1)
    fig.update_yaxes(title_text="Loss_total", row=2, col=2)

    fig.update_layout(
        title="Mean ± 1 std across seeds",
        height=900,
        width=1200,
        margin=dict(l=60, r=220, t=90, b=60),
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            orientation="v",
        ),
    )

    _apply_publication_style(fig)

    fig.write_image(cf.path_loss_accuracy_svg)
    fig.write_image(cf.path_loss_accuracy_png)
    return fig


class_name_map = {
    0: "crack",
    1: "cut",
    3: "hole",
    4: "print",
}


def plot_accuracy_across_seeds(dict_metrics_seed, cf):
    """
    Plots ONLY accuracy (mean ± 1 std across seeds).

    Left:  Train Accuracy
    Right: Test Accuracy
    """
    seeds = sorted(dict_metrics_seed.keys())

    def get_acc(seed, split):
        key = "acc_train" if split == "train" else "acc_test"
        return np.array(dict_metrics_seed[seed][split][key], dtype=float)

    acc_train = np.stack([get_acc(s, "train") for s in seeds], axis=0)
    acc_test = np.stack([get_acc(s, "test") for s in seeds], axis=0)

    epochs = np.arange(1, acc_train.shape[1] + 1)

    def mean_std(x):
        return x.mean(axis=0), x.std(axis=0)

    m_at, s_at = mean_std(acc_train)
    m_ate, s_ate = mean_std(acc_test)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Train Accuracy", "Test Accuracy"),
        horizontal_spacing=0.12,
    )

    def add_mean_std(row, col, mean, std, name, showlegend):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=mean,
                mode="lines+markers",
                name=name,
                showlegend=showlegend,
                customdata=np.stack([std], axis=-1),
                hovertemplate=(
                    "epoch=%{x}<br>"
                    "mean=%{y:.6f}<br>"
                    "std=%{customdata[0]:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([epochs, epochs[::-1]]),
                y=np.concatenate([mean + std, (mean - std)[::-1]]),
                fill="toself",
                line=dict(width=0),
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    add_mean_std(1, 1, m_at, s_at, "Train Accuracy (mean)", True)
    add_mean_std(1, 2, m_ate, s_ate, "Test Accuracy (mean)", True)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)

    fig.update_layout(
        title="Accuracy (mean ± 1 std across seeds)",
        height=500,
        width=1100,
        margin=dict(l=60, r=220, t=90, b=50),
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            orientation="v",
        ),
    )

    _apply_publication_style(fig)

    fig.write_image(cf.path_accuracy_plot_svg)
    fig.write_image(cf.path_accuracy_plot_png)
    return fig


def plot_winrate_boxplots(dict_win_rate_seed, cf, class_name_map=None):
    """
    2 rows:  Train (top), Test (bottom)
    Columns: overall + each class
    Each subplot: 3 boxplots (GC/SM/IG) with consistent colors + shared legend.
    """
    seeds = sorted(dict_win_rate_seed.keys())

    example_train = dict_win_rate_seed[seeds[0]]["train"]
    keys = ["overall"] + sorted([k for k in example_train.keys() if k != "overall"])

    def key_to_title(k):
        if k == "overall":
            return "overall"
        if class_name_map is not None and k in class_name_map:
            return class_name_map[k]
        return str(k)

    titles = [key_to_title(k) for k in keys]
    cols = len(keys)

    fig = make_subplots(
        rows=2,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.04,
        shared_yaxes=True,
    )

    methods = ["GC", "SM", "IG"]
    method_index = {"GC": 0, "SM": 1, "IG": 2}
    method_color = {"GC": "#1f77b4", "SM": "#ff7f0e", "IG": "#2ca02c"}

    def add_boxes(split, row, col, key):
        for m in methods:
            j = method_index[m]
            ys = [float(dict_win_rate_seed[s][split][key][j]) for s in seeds]
            fig.add_trace(
                go.Box(
                    y=ys,
                    name=m,
                    legendgroup=m,
                    marker_color=method_color[m],
                    boxmean=True,
                    showlegend=(row == 1 and col == 1),
                ),
                row=row,
                col=col,
            )

    for col_i, key in enumerate(keys, start=1):
        add_boxes("train", row=1, col=col_i, key=key)
        add_boxes("test", row=2, col=col_i, key=key)

    fig.update_layout(
        height=750,
        width=max(1100, 230 * cols),
        title="Win-rate distribution across seeds",
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            orientation="v",
        ),
        margin=dict(l=40, r=220, t=90, b=40),
    )

    # y-axis ranges
    fig.update_yaxes(title_text="win_rate (P(rank=1))", range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1])

    # Row labels
    fig.add_annotation(
        text="Train",
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14),
    )
    fig.add_annotation(
        text="Test",
        x=0.5,
        y=0.48,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14),
    )

    _apply_publication_style(fig)

    fig.write_image(cf.path_win_rate_svg)
    fig.write_image(cf.path_win_rate_png)
    return fig


def print_mean_std_metrics(dict_metrics_seed, dict_win_rate_seed, class_name_map=None):
    """
    Prints:
    1) Mean ± std per epoch for:
       - accuracy
       - loss_total
       - loss_ce
       - loss_bce
       (train & test)
    2) Win-rate mean ± std for each class and overall

    Expected structure:
    dict_metrics_seed[seed]["train"]["loss_train"] -> [[loss_total, loss_ce, loss_bce], ...]
    dict_metrics_seed[seed]["train"]["acc_train"]  -> [acc_epoch_1, ...]
    dict_win_rate_seed[seed]["train"][class_key]   -> [GC, SM, IG]
    """

    seeds = sorted(dict_metrics_seed.keys())

    print("=" * 80)
    print("METRICS ACROSS SEEDS (mean ± std)")
    print("=" * 80)

    # -------------------------
    # 1. LOSS & ACCURACY
    # -------------------------
    for split in ["train", "test"]:
        print(f"\n[{split.upper()}]")

        acc_key = "acc_train" if split == "train" else "acc_test"
        loss_key = "loss_train" if split == "train" else "loss_test"

        # Accuracy: (S, E)
        acc = np.stack([dict_metrics_seed[s][split][acc_key] for s in seeds], axis=0)

        # Losses: (S, E)
        loss_total = np.stack(
            [[x[0] for x in dict_metrics_seed[s][split][loss_key]] for s in seeds],
            axis=0,
        )
        loss_ce = np.stack(
            [[x[1] for x in dict_metrics_seed[s][split][loss_key]] for s in seeds],
            axis=0,
        )
        loss_bce = np.stack(
            [[x[2] for x in dict_metrics_seed[s][split][loss_key]] for s in seeds],
            axis=0,
        )

        # Mean ± std
        acc_mean, acc_std = acc.mean(axis=0), acc.std(axis=0)
        lt_mean, lt_std = loss_total.mean(axis=0), loss_total.std(axis=0)
        lce_mean, lce_std = loss_ce.mean(axis=0), loss_ce.std(axis=0)
        lbce_mean, lbce_std = loss_bce.mean(axis=0), loss_bce.std(axis=0)

        for ep in range(acc.shape[1]):
            print(
                f"Epoch {ep + 1:02d} | "
                f"Acc: {acc_mean[ep]:.4f} ± {acc_std[ep]:.4f} | "
                f"Loss_total: {lt_mean[ep]:.4f} ± {lt_std[ep]:.4f} | "
                f"Loss_CE: {lce_mean[ep]:.4f} ± {lce_std[ep]:.4f} | "
                f"Loss_BCE: {lbce_mean[ep]:.4f} ± {lbce_std[ep]:.4f}"
            )

    # -------------------------
    # 2. WIN RATE
    # -------------------------
    print("\n" + "=" * 80)
    print("WIN-RATE ACROSS SEEDS (mean ± std)")
    print("=" * 80)

    for split in ["train", "test"]:
        print(f"\n[{split.upper()}]")

        example = dict_win_rate_seed[seeds[0]][split]
        keys = ["overall"] + sorted([k for k in example.keys() if k != "overall"])

        for k in keys:
            win_rates = np.stack(
                [dict_win_rate_seed[s][split][k] for s in seeds], axis=0
            )  # (S, 3)

            mean = win_rates.mean(axis=0)
            std = win_rates.std(axis=0)

            if k == "overall":
                name = "overall"
            else:
                name = (
                    class_name_map[k]
                    if class_name_map and k in class_name_map
                    else str(k)
                )

            print(f"\nClass: {name}")
            print(
                f"  GC: {mean[0]:.4f} ± {std[0]:.4f} | "
                f"SM: {mean[1]:.4f} ± {std[1]:.4f} | "
                f"IG: {mean[2]:.4f} ± {std[2]:.4f}"
            )
