import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass
from typing import Tuple

# --- 1. Data Models & Logic Layer ---


@dataclass
class TwoSampleTestParameters:
    """Data Transfer Object for two-sample test parameters."""

    mu1: float
    s1: float
    n1: int  # Group 1 stats
    mu2: float
    s2: float
    n2: int  # Group 2 stats
    alpha: float  # Significance Level


class PooledTTestCalculator:
    """
    Handles the statistical logic for a Two-Sample Pooled T-Test.
    (Assumes equal variances).
    """

    def __init__(self, params: TwoSampleTestParameters):
        self.p = params
        self.df = self.p.n1 + self.p.n2 - 2
        self.mean_diff = self.p.mu1 - self.p.mu2

        self.sp_sq = self._calculate_pooled_variance()
        self.se = self._calculate_standard_error()
        self.t_score = self._calculate_t_score()
        self.t_critical = self._calculate_critical_t()

    def _calculate_pooled_variance(self) -> float:
        num = (self.p.n1 - 1) * self.p.s1**2 + (self.p.n2 - 1) * self.p.s2**2
        den = self.p.n1 + self.p.n2 - 2
        return num / den

    def _calculate_standard_error(self) -> float:
        # Standard Error of the difference between means
        return np.sqrt(self.sp_sq * (1 / self.p.n1 + 1 / self.p.n2))

    def _calculate_t_score(self) -> float:
        # Testing against Null Hypothesis: diff = 0
        return (self.mean_diff - 0) / self.se

    def _calculate_critical_t(self) -> float:
        return stats.t.ppf(1 - self.p.alpha / 2, self.df)

    def get_p_value(self) -> float:
        return 2 * (1 - stats.t.cdf(abs(self.t_score), self.df))

    def get_critical_region_bounds(self) -> Tuple[float, float]:
        """Returns the raw difference values where rejection regions start."""
        margin_of_error = self.t_critical * self.se
        # Centered at 0 (Null Hypothesis)
        return (0 - margin_of_error, 0 + margin_of_error)

    def is_rejected(self) -> bool:
        return abs(self.t_score) > self.t_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    def __init__(self):
        st.set_page_config(
            page_title="Algorithm Comparison Test", layout="centered", page_icon="⚖️"
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
        calculator = PooledTTestCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("⚖️ Algorithm Throughput Comparison")
        st.markdown("""
        **Scenario:** Comparing two load-balancing algorithms (Group A vs Group B).
        * **Metric:** Throughput (TPS).
        * **Null Hypothesis ($H_0$):** $\mu_A - \mu_B = 0$ (No difference).
        * **Alternative ($H_1$):** $\mu_A - \mu_B \\neq 0$ (Significant difference).
        """)
        st.markdown("---")

    def _render_sidebar(self) -> TwoSampleTestParameters:
        st.sidebar.header("1. Group A (Round-Robin)")
        mu1 = st.sidebar.number_input(
            "Mean Throughput ($\overline{x}_1$)", value=24.7, step=0.1
        )
        s1 = st.sidebar.number_input("Std Deviation ($s_1$)", value=1.55, step=0.01)
        n1 = st.sidebar.number_input("Sample Size ($n_1$)", value=100, step=1)

        st.sidebar.markdown("---")

        st.sidebar.header("2. Group B (Least-Conn)")
        mu2 = st.sidebar.number_input(
            "Mean Throughput ($\overline{x}_2$)", value=24.2, step=0.1
        )
        s2 = st.sidebar.number_input("Std Deviation ($s_2$)", value=1.70, step=0.01)
        n2 = st.sidebar.number_input("Sample Size ($n_2$)", value=100, step=1)

        st.sidebar.markdown("---")
        st.sidebar.header("3. Settings")
        alpha = st.sidebar.selectbox(
            "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
        )

        return TwoSampleTestParameters(mu1, s1, int(n1), mu2, s2, int(n2), float(alpha))

    def _render_visualization(self, calc: PooledTTestCalculator):
        """
        Plots the Sampling Distribution of the DIFFERENCE between means.
        """
        # We plot centered at 0 (H0)
        center = 0
        se = calc.se
        df = calc.df
        x_left_crit, x_right_crit = calc.get_critical_region_bounds()

        # Define x range (Standardized differences)
        x = np.linspace(center - 4.5 * se, center + 4.5 * se, 1000)
        y = stats.t.pdf(x, df, loc=center, scale=se)

        # Fill Data
        x_fill_left = np.linspace(x[0], x_left_crit, 100)
        y_fill_left = stats.t.pdf(x_fill_left, df, loc=center, scale=se)

        x_fill_right = np.linspace(x_right_crit, x[-1], 100)
        y_fill_right = stats.t.pdf(x_fill_right, df, loc=center, scale=se)

        x_fill_accept = np.linspace(x_left_crit, x_right_crit, 100)
        y_fill_accept = stats.t.pdf(x_fill_accept, df, loc=center, scale=se)

        fig = go.Figure()

        # 1. Rejection Regions
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

        # 2. Acceptance Region
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

        # 3. Main Curve
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=f"Dist. of Differences (df={df})",
                mode="lines",
                opacity=0.8,
                line=dict(color=self.colors["dist_line"], width=2),
            )
        )

        # 4. Observed Statistic
        fig.add_vline(
            x=calc.mean_diff,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"Observed Diff: {calc.mean_diff:.2f}",
            annotation_position="top right",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        fig.update_layout(
            title=dict(
                text=f"Distribution of Differences (Confidence = {1 - calc.p.alpha:.2f})",
                font=dict(size=18),
            ),
            xaxis_title=dict(text="Difference in Means (TPS)", font=dict(size=16)),
            yaxis_title=dict(text="Density", font=dict(size=16)),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=14)
            ),
            template="plotly_white",
            height=600,
            margin=dict(t=50, b=50, l=50, r=50),
        )

        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "algo_comparison_result",
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

    def _render_results(self, calc: PooledTTestCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)

        col1.metric("t-Score", f"{calc.t_score:.3f}")
        col2.metric("Critical t Threshold", f"±{calc.t_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Throughput difference is statistically significant."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "Performance difference is indistinguishable from noise."
            delta_color = "normal"

        col3.metric(
            "Conclusion",
            decision_text,
            f"p = {calc.get_p_value():.4f}",
            delta_color=delta_color,
        )

        if calc.is_rejected():
            st.error(
                f"**Action Required:** {detail_text} One algorithm is clearly superior."
            )
        else:
            st.success(f"**Status:** {detail_text} No clear winner based on this data.")


# --- 3. Entry Point ---
if __name__ == "__main__":
    app = StreamlitHypothesisApp()
    app.run()
