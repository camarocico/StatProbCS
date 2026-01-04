import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass
from typing import Tuple

# --- 1. Data Models & Logic Layer ---


@dataclass
class TwoProportionTestParameters:
    """Data Transfer Object for A/B test parameters."""

    n1: int
    x1: int  # Group A (Sample size, Successes)
    n2: int
    x2: int  # Group B
    alpha: float  # Significance Level


class TwoProportionZTestCalculator:
    """
    Handles statistical logic for Two-Sample Z-Test for Proportions.
    """

    def __init__(self, params: TwoProportionTestParameters):
        self.p = params

        # Sample Proportions
        self.p1_hat = self.p.x1 / self.p.n1
        self.p2_hat = self.p.x2 / self.p.n2
        self.diff = self.p1_hat - self.p2_hat

        self.se = self._calculate_standard_error()
        self.z_score = self._calculate_z_score()
        self.z_critical = self._calculate_critical_z()

    def _calculate_standard_error(self) -> float:
        # Pooled Proportion
        total_success = self.p.x1 + self.p.x2
        total_n = self.p.n1 + self.p.n2
        p_pool = total_success / total_n

        # Standard Error formula for difference of proportions
        return np.sqrt(p_pool * (1 - p_pool) * (1 / self.p.n1 + 1 / self.p.n2))

    def _calculate_z_score(self) -> float:
        return (self.diff - 0) / self.se

    def _calculate_critical_z(self) -> float:
        return stats.norm.ppf(1 - self.p.alpha / 2)

    def get_p_value(self) -> float:
        return 2 * (1 - stats.norm.cdf(abs(self.z_score)))

    def get_critical_region_bounds(self) -> Tuple[float, float]:
        """Returns the raw difference values where rejection regions start."""
        margin_of_error = self.z_critical * self.se
        return (0 - margin_of_error, 0 + margin_of_error)

    def is_rejected(self) -> bool:
        return abs(self.z_score) > self.z_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    def __init__(self):
        st.set_page_config(
            page_title="A/B Testing Calculator", layout="centered", page_icon="🛒"
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
        calculator = TwoProportionZTestCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("🛒 UX A/B Testing Calculator")
        st.markdown("""
        **Scenario:** Comparing conversion rates of two designs (A vs B).
        * **Metric:** Difference in Proportions ($p_A - p_B$).
        * **Null Hypothesis ($H_0$):** $p_A - p_B = 0$ (No difference).
        * **Alternative ($H_1$):** $p_A - p_B \\neq 0$ (Significant difference).
        """)
        st.markdown("---")

    def _render_sidebar(self) -> TwoProportionTestParameters:
        st.sidebar.header("1. Design A (Control)")
        n1 = st.sidebar.number_input("Visitors ($n_A$)", value=1000, step=10)
        x1 = st.sidebar.number_input("Conversions ($x_A$)", value=520, step=1)
        st.sidebar.caption(f"Conversion Rate: **{x1 / n1:.2%}**")

        st.sidebar.markdown("---")

        st.sidebar.header("2. Design B (Variant)")
        n2 = st.sidebar.number_input("Visitors ($n_B$)", value=950, step=10)
        x2 = st.sidebar.number_input("Conversions ($x_B$)", value=480, step=1)
        st.sidebar.caption(f"Conversion Rate: **{x2 / n2:.2%}**")

        st.sidebar.markdown("---")
        st.sidebar.header("3. Settings")
        alpha = st.sidebar.selectbox(
            "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
        )

        return TwoProportionTestParameters(
            int(n1), int(x1), int(n2), int(x2), float(alpha)
        )

    def _render_visualization(self, calc: TwoProportionZTestCalculator):
        """
        Plots the Normal Distribution of the DIFFERENCE in proportions.
        """
        center = 0  # Null Hypothesis
        se = calc.se
        x_left_crit, x_right_crit = calc.get_critical_region_bounds()

        # Define x range (Standardized differences)
        x = np.linspace(center - 4.5 * se, center + 4.5 * se, 1000)
        y = stats.norm.pdf(x, center, se)

        # Fill Data
        x_fill_left = np.linspace(x[0], x_left_crit, 100)
        y_fill_left = stats.norm.pdf(x_fill_left, center, se)

        x_fill_right = np.linspace(x_right_crit, x[-1], 100)
        y_fill_right = stats.norm.pdf(x_fill_right, center, se)

        x_fill_accept = np.linspace(x_left_crit, x_right_crit, 100)
        y_fill_accept = stats.norm.pdf(x_fill_accept, center, se)

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
                name="Dist. of Differences",
                mode="lines",
                opacity=0.8,
                line=dict(color=self.colors["dist_line"], width=2),
            )
        )

        # 4. Observed Statistic
        fig.add_vline(
            x=calc.diff,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"Observed Diff: {calc.diff:.4f}",
            annotation_position="top right",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        fig.update_layout(
            title=dict(
                text=f"A/B Test Results (Confidence = {1 - calc.p.alpha:.2f})",
                font=dict(size=18),
            ),
            xaxis_title=dict(text="Difference in Proportions", font=dict(size=16)),
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
                "filename": "ab_test_result",
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

    def _render_results(self, calc: TwoProportionZTestCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)

        col1.metric("Z-Score", f"{calc.z_score:.3f}")
        col2.metric("Critical Z Threshold", f"±{calc.z_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Significant difference in conversion rates."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "Difference is not statistically significant."
            delta_color = "normal"

        col3.metric(
            "Conclusion",
            decision_text,
            f"p = {calc.get_p_value():.4f}",
            delta_color=delta_color,
        )

        if calc.is_rejected():
            st.error(f"**Action Required:** {detail_text} Deploy the winning design.")
        else:
            st.success(
                f"**Status:** {detail_text} Keep the current design or run test longer."
            )


# --- 3. Entry Point ---
if __name__ == "__main__":
    app = StreamlitHypothesisApp()
    app.run()
