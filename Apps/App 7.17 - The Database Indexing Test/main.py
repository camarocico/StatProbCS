import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass
from typing import Tuple

# --- 1. Data Models & Logic Layer ---


@dataclass
class PairedTestParameters:
    """Data Transfer Object for paired test parameters."""

    mu_diff_0: float  # Hypothesized Mean Difference (usually 0)
    mean_diff: float  # Observed Sample Mean Difference
    sd_diff: float  # Standard Deviation of Differences
    n: int  # Sample Size (Pairs)
    alpha: float  # Significance Level


class PairedTTestCalculator:
    """
    Handles the statistical logic for a Two-Tailed Paired T-Test.
    Mathematically equivalent to One-Sample T-Test on differences.
    """

    def __init__(self, params: PairedTestParameters):
        self.params = params
        self.df = params.n - 1
        self.se = self._calculate_standard_error()
        self.t_score = self._calculate_t_score()
        self.t_critical = self._calculate_critical_t()

    def _calculate_standard_error(self) -> float:
        return self.params.sd_diff / np.sqrt(self.params.n)

    def _calculate_t_score(self) -> float:
        return (self.params.mean_diff - self.params.mu_diff_0) / self.se

    def _calculate_critical_t(self) -> float:
        # Two-tailed critical value
        return stats.t.ppf(1 - self.params.alpha / 2, self.df)

    def get_p_value(self) -> float:
        return 2 * (1 - stats.t.cdf(abs(self.t_score), self.df))

    def get_critical_region_bounds(self) -> Tuple[float, float]:
        """Returns the raw X values (differences) where rejection regions start."""
        margin_of_error = self.t_critical * self.se
        return (
            self.params.mu_diff_0 - margin_of_error,
            self.params.mu_diff_0 + margin_of_error,
        )

    def is_rejected(self) -> bool:
        return abs(self.t_score) > self.t_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    def __init__(self):
        st.set_page_config(
            page_title="DB Optimization Paired Test", layout="centered", page_icon="📉"
        )
        self.colors = {
            "rejection_fill": "salmon",
            "rejection_line": "red",
            "accept_fill": "darkcyan",
            "accept_line": "green",
            "dist_line": "blue",
            "statistic": "purple",
        }

    def run(self):
        self._render_header()
        params = self._render_sidebar()
        calculator = PairedTTestCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("📉 DB Indexing Paired T-Test")
        st.markdown("""
        **Scenario:** Testing if a database index changes query speed.
        * **Metric:** Difference in time ($d = \\textrm{Time}_{\\textrm{after}} - \\textrm{Time}_{\\textrm{before}}$).
        * **Null Hypothesis ($H_0$):** $\mu_d = 0$ (No effect).
        * **Alternative ($H_1$):** $\mu_d \\neq 0$ (Significant change).
        """)
        st.markdown("---")

    def _render_sidebar(self) -> PairedTestParameters:
        st.sidebar.header("1. Hypothesized Difference ($H_0$)")
        # Usually 0 for "No Effect"
        mu_diff_0 = st.sidebar.number_input(
            "Expected Difference (ms)", value=0.0, disabled=True
        )

        st.sidebar.header("2. Sample Data (Differences)")
        n = st.sidebar.number_input("Sample Size (Pairs)", value=30, step=1)

        # User inputs
        sd_diff = st.sidebar.slider(
            "Std Dev of Differences ($s_d$)",
            min_value=10.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
        )
        st.sidebar.caption(f"Standard Error: **{sd_diff / np.sqrt(n):.2f}**")

        mean_diff = st.sidebar.slider(
            "Mean Difference ($\overline{x}_d$)",
            min_value=-50.0,
            max_value=50.0,
            value=-21.0,
            step=1.0,
        )

        st.sidebar.header("3. Significance")
        alpha = st.sidebar.selectbox("Alpha ($\\alpha$)", [0.01, 0.05, 0.10], index=1)

        return PairedTestParameters(mu_diff_0, mean_diff, sd_diff, int(n), float(alpha))

    def _render_visualization(self, calc: PairedTTestCalculator):
        """
        Plots the T-Distribution of the Differences.
        """
        mu = calc.params.mu_diff_0
        se = calc.se
        df = calc.df
        x_left_crit, x_right_crit = calc.get_critical_region_bounds()

        # Define x range (centered on 0, wide enough to show tails)
        # We generally want 4 standard errors to left and right
        x = np.linspace(mu - 4.5 * se, mu + 4.5 * se, 1000)
        y = stats.t.pdf(x, df, loc=mu, scale=se)

        # Define Fill Regions
        x_fill_left = np.linspace(x[0], x_left_crit, 100)
        y_fill_left = stats.t.pdf(x_fill_left, df, loc=mu, scale=se)

        x_fill_right = np.linspace(x_right_crit, x[-1], 100)
        y_fill_right = stats.t.pdf(x_fill_right, df, loc=mu, scale=se)

        x_fill_accept = np.linspace(x_left_crit, x_right_crit, 100)
        y_fill_accept = stats.t.pdf(x_fill_accept, df, loc=mu, scale=se)

        fig = go.Figure()

        # 1. Left Rejection Region
        fig.add_trace(
            go.Scatter(
                x=x_fill_left,
                y=y_fill_left,
                fill="tozeroy",
                fillcolor=self.colors["rejection_fill"],
                name="Left Rejection Region",
                mode="lines",
                line=dict(color=self.colors["rejection_line"], width=2),
            )
        )

        # 2. Right Rejection Region
        fig.add_trace(
            go.Scatter(
                x=x_fill_right,
                y=y_fill_right,
                fill="tozeroy",
                fillcolor=self.colors["rejection_fill"],
                name="Right Rejection Region",
                mode="lines",
                line=dict(color=self.colors["rejection_line"], width=2),
            )
        )

        # 3. Acceptance Region
        fig.add_trace(
            go.Scatter(
                x=x_fill_accept,
                y=y_fill_accept,
                fill="tozeroy",
                fillcolor=self.colors["accept_fill"],
                name="Non-Rejection Region",
                mode="lines",
                line=dict(color=self.colors["accept_line"], width=2),
            )
        )

        # 4. Distribution Outline
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=f"T-Distribution of Differences (df={df})",
                mode="lines",
                opacity=0.8,
                line=dict(color=self.colors["dist_line"], width=2),
            )
        )

        # 5. Statistic Line
        fig.add_vline(
            x=calc.params.mean_diff,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"Observed Diff: {calc.params.mean_diff:.1f} ms",
            annotation_position="top left",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        fig.update_layout(
            title=dict(
                text=f"Paired T-Test Distribution (Confidence = {1 - calc.params.alpha:.2f})",
                font=dict(size=18),
            ),
            xaxis_title=dict(text="Mean Difference (ms)", font=dict(size=16)),
            yaxis_title=dict(text="Density", font=dict(size=16)),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.6, font=dict(size=14)
            ),
            template="plotly_white",
            height=600,
            margin=dict(t=50, b=50, l=50, r=50),
        )

        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "paired_test_result",
                "height": 1080,
                "width": 1920,
                "scale": 2,
            },
            "displayModeBar": True,
        }

        st.plotly_chart(fig, use_container_width=True, config=config)
        st.caption(
            "📷 **Tip:** Hover over the chart and click the **camera icon** to download a High-Res PNG."
        )

    def _render_results(self, calc: PairedTTestCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)

        col1.metric("t-Score", f"{calc.t_score:.3f}")
        col2.metric("Critical t Threshold", f"±{calc.t_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Significant effect detected."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "No statistically significant effect found."
            delta_color = "normal"

        col3.metric(
            "Conclusion",
            decision_text,
            f"p = {calc.get_p_value():.4f}",
            delta_color=delta_color,
        )

        if calc.is_rejected():
            st.error(
                f"**Action Required:** {detail_text} The index changes performance."
            )
        else:
            st.success(
                f"**Status:** {detail_text} The index impact is indistinguishable from noise."
            )


# --- 3. Entry Point ---
if __name__ == "__main__":
    app = StreamlitHypothesisApp()
    app.run()
