import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


def get_z_score(confidence_level: float = 0.95) -> float:
    """
    Calculates the Z-score for a two-tailed confidence interval.
    """
    alpha = 1 - confidence_level
    return norm.ppf(1 - alpha / 2)


def simulate_draws(red_fraction: float, num_draws: int) -> np.ndarray:
    """
    Simulates drawing cards and returns the count of reds.
    """
    num_reds_drawn = np.random.binomial(num_draws, red_fraction)
    return np.array([1] * num_reds_drawn + [0] * (num_draws - num_reds_drawn))


def calculate_ci_stats(data, red_fraction, confidence_level):
    """Calculates mean, bounds, and whether the CI captures the truth."""
    mean = np.mean(data)

    if mean == 0 or mean == 1:
        std_dev = 0
    else:
        std_dev = np.sqrt(mean * (1 - mean))

    z = get_z_score(confidence_level)
    margin_error = z * (std_dev / np.sqrt(len(data)))

    lower = mean - margin_error
    upper = mean + margin_error

    captured = not ((lower > red_fraction) or (upper < red_fraction))

    return {
        "mean": mean,
        "width": upper - lower,
        "lower": lower,
        "upper": upper,
        "captured": captured,
    }


def get_theoretical_width(p, n, confidence_level):
    """Calculates the expected CI width for a given p, n, and confidence level."""
    if n == 0:
        return 0
    z = get_z_score(confidence_level)
    return 2 * z * np.sqrt(p * (1 - p) / n)


# --- Plotting Functions ---


def plot_accumulated_results(results, red_fraction, confidence_level):
    """Tab 1: Plots the history of basic interactive results."""
    if not results:
        return go.Figure()

    means = [r["mean"] for r in results]
    lowers = [r["lower"] for r in results]
    uppers = [r["upper"] for r in results]

    colors = ["darkcyan" if r["captured"] else "red" for r in results]
    status_text = [
        "Captured Truth" if r["captured"] else "Missed Truth" for r in results
    ]

    error_y_upper = np.array(uppers) - np.array(means)
    error_y_lower = np.array(means) - np.array(lowers)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(results) + 1)),
            y=means,
            mode="markers",
            name="Simulation",
            marker=dict(color=colors, size=8),
            error_y=dict(
                type="data",
                symmetric=False,
                array=error_y_upper,
                arrayminus=error_y_lower,
                color="rgba(100,100,100,0.5)",
            ),
            hovertemplate="<b>Sim #%{x}</b><br>Est: %{y}<br>%{text}<extra></extra>",
            text=status_text,
        )
    )

    fig.add_hline(
        y=red_fraction,
        line_dash="dash",
        line_color="green",
        annotation_text="True Fraction",
    )

    fig.update_layout(
        title=f"Building {confidence_level * 100:.0f}% Confidence Intervals",
        xaxis_title="Simulation Sequence",
        yaxis_title="Estimated Red Fraction",
        yaxis_range=[0, 1],
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_size_experiments(experiments, red_fraction, confidence_level):
    """Tab 2: Plots user-run experiments vs Theoretical Curve."""
    fig = go.Figure()

    # 1. Theoretical Curve
    n_theory = np.linspace(10, 1000, 100)
    w_theory = [
        get_theoretical_width(red_fraction, n, confidence_level) for n in n_theory
    ]

    fig.add_trace(
        go.Scatter(
            x=n_theory,
            y=w_theory,
            mode="lines",
            name="Theoretical Width",
            line=dict(color="gray", dash="dash"),
            hoverinfo="skip",
        )
    )

    # 2. User Experiment Data
    if experiments:
        ns = [e["n"] for e in experiments]
        widths = [e["avg_width"] for e in experiments]

        fig.add_trace(
            go.Scatter(
                x=ns,
                y=widths,
                mode="markers",
                name="Measured Avg Width",
                marker=dict(
                    color="darkcyan", size=10, line=dict(width=2, color="white")
                ),
                hovertemplate="<b>n = %{x}</b><br>Avg Width: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Sample Size vs. Width ({confidence_level * 100:.0f}% CI)",
        xaxis_title="Sample Size (n)",
        yaxis_title="Average CI Width",
        xaxis_type="log",
        yaxis_range=[0, 1.0],
        height=500,
    )
    return fig


# --- Main Application ---


def main():
    st.set_page_config(page_title="Interactive CI Builder", layout="wide")
    st.title("📊 Interactive Confidence Interval Builder")

    # --- Sidebar Global Settings ---
    st.sidebar.header("Global Settings")

    # Session state initialization
    if "global_p" not in st.session_state:
        st.session_state["global_p"] = 0.60
    if "global_cl" not in st.session_state:
        st.session_state["global_cl"] = 0.95

    # 1. Population Parameter
    red_fraction = st.sidebar.slider(
        "True Red Fraction ($p$)", 0.0, 1.0, st.session_state["global_p"], 0.01
    )

    # 2. Confidence Level Parameter
    confidence_level = st.sidebar.slider(
        "Confidence Level ($1 - \\alpha$)",
        min_value=0.80,
        max_value=0.99,
        value=st.session_state["global_cl"],
        step=0.01,
        format="%.2f",
    )

    # Reset history if p or confidence level changes
    if (
        red_fraction != st.session_state["global_p"]
        or confidence_level != st.session_state["global_cl"]
    ):
        st.session_state["history"] = []
        st.session_state["size_experiments"] = []
        st.session_state["global_p"] = red_fraction
        st.session_state["global_cl"] = confidence_level

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Z-score:** `{get_z_score(confidence_level):.3f}`")

    # --- Tabs ---
    tab1, tab2 = st.tabs(
        ["1. Construct CIs (Randomness)", "2. Investigate Sample Size (Precision)"]
    )

    # ==========================================
    # TAB 1: Construct CIs
    # ==========================================
    with tab1:
        st.subheader("Visualizing Random Variation")
        st.markdown(
            f"Current Settings: **{confidence_level * 100:.0f}% Confidence Level**"
        )

        # Local Sample Size Control for Tab 1
        if "n_tab1" not in st.session_state:
            st.session_state["n_tab1"] = 27
        num_draws = st.number_input(
            "Sample Size for these CIs ($n$)",
            1,
            1000,
            st.session_state["n_tab1"],
            1,
            key="n_input_tab1",
        )

        if num_draws != st.session_state["n_tab1"]:
            st.session_state["history"] = []
            st.session_state["n_tab1"] = num_draws

        # History Init
        if "history" not in st.session_state:
            st.session_state["history"] = []

        # Controls
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        with c1:
            if st.button("Draw 1 Sample", key="draw1"):
                data = simulate_draws(red_fraction, num_draws)
                st.session_state["history"].append(
                    calculate_ci_stats(data, red_fraction, confidence_level)
                )
        with c2:
            if st.button("Draw 10 Samples", key="draw10"):
                for _ in range(10):
                    data = simulate_draws(red_fraction, num_draws)
                    st.session_state["history"].append(
                        calculate_ci_stats(data, red_fraction, confidence_level)
                    )
        with c3:
            if st.button("Draw 100 Samples", key="draw100"):
                for _ in range(100):
                    data = simulate_draws(red_fraction, num_draws)
                    st.session_state["history"].append(
                        calculate_ci_stats(data, red_fraction, confidence_level)
                    )
        with c5:
            if st.button("Clear Plot", key="clear1"):
                st.session_state["history"] = []

        # Metrics & Plot
        if st.session_state["history"]:
            total = len(st.session_state["history"])
            bad = sum(1 for r in st.session_state["history"] if not r["captured"])
            pct = (bad / total) * 100

            target_error = (1 - confidence_level) * 100
            st.metric(
                "Coverage Error Rate",
                f"{pct:.1f}%",
                delta=f"{pct - target_error:.1f}% vs target ({target_error:.1f}%)",
                delta_color="inverse",
            )

            fig = plot_accumulated_results(
                st.session_state["history"], red_fraction, confidence_level
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("👈 Set parameters and click 'Draw' to start.")

    # ==========================================
    # TAB 2: Investigate Sample Size
    # ==========================================
    with tab2:
        st.subheader("Discovering the Relationship")
        st.markdown(
            f"Investigating width for **{confidence_level * 100:.0f}% Confidence Level**."
        )

        if "size_experiments" not in st.session_state:
            st.session_state["size_experiments"] = []

        col_input, col_action = st.columns([2, 1])
        with col_input:
            n_to_test = st.slider(
                "Select Sample Size to Test ($n$)",
                1,
                1000,
                10,
                1,
                key="n_slider_tab2",
            )

        with col_action:
            st.write("##")
            if st.button(f"Run Experiment (n={n_to_test})", width="stretch"):
                batch_size = 50
                batch_widths = []
                for _ in range(batch_size):
                    data = simulate_draws(red_fraction, n_to_test)
                    stats = calculate_ci_stats(data, red_fraction, confidence_level)
                    batch_widths.append(stats["width"])

                avg_width = np.mean(batch_widths)
                st.session_state["size_experiments"].append(
                    {"n": n_to_test, "avg_width": avg_width}
                )

        if st.button("Clear Experiments", key="clear2"):
            st.session_state["size_experiments"] = []

        fig_theory = plot_size_experiments(
            st.session_state["size_experiments"], red_fraction, confidence_level
        )
        st.plotly_chart(fig_theory, width="stretch")

        if st.session_state["size_experiments"]:
            st.info(
                "Try increasing the Confidence Level in the sidebar (e.g., to 0.99). You will see the Theoretical Curve shift upwards because we need wider intervals to be more confident."
            )


if __name__ == "__main__":
    main()
