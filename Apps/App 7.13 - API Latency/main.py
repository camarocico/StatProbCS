import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass
from typing import Tuple

# --- 1. Data Models & Logic Layer ---


@dataclass
class TestParameters:
    """Data Transfer Object for test parameters."""

    mu_0: float  # Population Mean (Hypothesized)
    sample_mean: float  # Observed Sample Mean
    sigma: float  # Population Standard Deviation
    n: int  # Sample Size
    alpha: float  # Significance Level


class ZTestCalculator:
    """
    Handles the statistical logic for a two-tailed Z-test.
    Independent of UI.
    """

    def __init__(self, params: TestParameters):
        self.params = params
        self.se = self._calculate_standard_error()
        self.z_score = self._calculate_z_score()
        self.z_critical = self._calculate_critical_z()

    def _calculate_standard_error(self) -> float:
        return self.params.sigma / np.sqrt(self.params.n)

    def _calculate_z_score(self) -> float:
        return (self.params.sample_mean - self.params.mu_0) / self.se

    def _calculate_critical_z(self) -> float:
        return stats.norm.ppf(1 - self.params.alpha / 2)

    def get_p_value(self) -> float:
        return 2 * (1 - stats.norm.cdf(abs(self.z_score)))

    def get_critical_region_bounds(self) -> Tuple[float, float]:
        """Returns the raw X values where rejection regions start."""
        margin_of_error = self.z_critical * self.se
        return (self.params.mu_0 - margin_of_error, self.params.mu_0 + margin_of_error)

    def is_rejected(self) -> bool:
        return abs(self.z_score) > self.z_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    """
    Manages the Streamlit User Interface.
    """

    def __init__(self):
        st.set_page_config(
            page_title="DevOps Hypothesis Test",
            layout="centered",
            page_icon="Fig_07_01",
        )
        # Colors extracted from your Fig_07_01.py example
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
        calculator = ZTestCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("🖥️ API Latency Hypothesis Tester")
        st.markdown("""
        **Scenario:** Determine if API latency has significantly deviated from the SLA.
        * **Null Hypothesis ($H_0$):** $\mu = 50$ ms.
        * **Alternative ($H_1$):** $\mu \\neq 50$ ms.
        """)
        st.markdown("---")

    def _render_sidebar(self) -> TestParameters:
        st.sidebar.header("1. Configuration")
        with st.sidebar.expander("SLA Definitions ($H_0$)", expanded=True):
            mu_0 = st.number_input("Target Latency (ms)", value=50.0, step=0.1)
            n = st.number_input("Sample Size (n)", value=64, step=1)
            alpha = st.selectbox(
                "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
            )

        st.sidebar.header("2. Observations")
        initial_var = 64.0
        sigma_sq = st.sidebar.slider(
            "Population Variance ($\sigma^2$)",
            min_value=1.0,
            max_value=150.0,
            value=initial_var,
            step=0.5,
        )
        sigma = np.sqrt(sigma_sq)
        st.sidebar.caption(f"Implied Std Dev ($\sigma$): **{sigma:.2f} ms**")

        sample_mean = st.sidebar.slider(
            "Observed Sample Mean (ms)",
            min_value=40.0,
            max_value=60.0,
            value=53.0,
            step=0.1,
        )
        return TestParameters(mu_0, sample_mean, sigma, int(n), float(alpha))

    def _render_visualization(self, calc: ZTestCalculator):
        """
        Plots the distribution matching the style of Fig_07_01.py
        """
        mu = calc.params.mu_0
        se = calc.se
        x_left_crit, x_right_crit = calc.get_critical_region_bounds()

        # Define x range for the plot
        x = np.linspace(mu - 4 * se, mu + 4 * se, 1000)
        y = stats.norm.pdf(x, mu, se)

        # Define specific regions for shading (matching the example logic)
        x_fill_left = np.linspace(x[0], x_left_crit, 100)
        y_fill_left = stats.norm.pdf(x_fill_left, mu, se)

        x_fill_right = np.linspace(x_right_crit, x[-1], 100)
        y_fill_right = stats.norm.pdf(x_fill_right, mu, se)

        x_fill_acceptance = np.linspace(x_left_crit, x_right_crit, 100)
        y_fill_acceptance = stats.norm.pdf(x_fill_acceptance, mu, se)

        fig = go.Figure()

        # 1. Left Rejection Region (Salmon)
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

        # 2. Right Rejection Region (Salmon)
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

        # 3. Acceptance Region (Dark Cyan)
        fig.add_trace(
            go.Scatter(
                x=x_fill_acceptance,
                y=y_fill_acceptance,
                fill="tozeroy",
                fillcolor=self.colors["accept_fill"],
                name="Non-Rejection Region",
                mode="lines",
                line=dict(color=self.colors["accept_line"], width=2),
            )
        )

        # 4. The Standard Normal Line (Blue)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name="Normal Distribution",
                mode="lines",
                opacity=0.8,
                line=dict(color=self.colors["dist_line"], width=2),
            )
        )

        # 5. The Observed Statistic (Purple Dashed Line)
        # Using add_vline to match the provided example style
        fig.add_vline(
            x=calc.params.sample_mean,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"Observed Mean = {calc.params.sample_mean:.1f}",
            annotation_position="top left",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        # Layout styling matching the reference
        fig.update_layout(
            title=dict(
                text=f"Rejection and Acceptance Regions (Confidence level = {1 - calc.params.alpha:.2f})",
                font=dict(size=18),
            ),
            xaxis_title=dict(text="Latency (ms)", font=dict(size=16)),
            yaxis_title=dict(text="Density", font=dict(size=16)),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=14)
            ),
            template="plotly_white",
            height=600,
            margin=dict(t=50, b=50, l=50, r=50),
        )

        # --- High-Res Download Configuration ---
        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "hypothesis_test_result",
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

    def _render_results(self, calc: ZTestCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)
        col1.metric("Z-Score", f"{calc.z_score:.3f}")
        col2.metric("Critical Z Threshold", f"±{calc.z_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Anomaly Detected: Latency deviation is significant."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "System Normal: Deviation is within random variance."
            delta_color = "normal"

        col3.metric(
            "Conclusion",
            decision_text,
            f"p = {calc.get_p_value():.4f}",
            delta_color=delta_color,
        )

        if calc.is_rejected():
            st.error(f"**Action Required:** {detail_text}")
        else:
            st.success(f"**Status:** {detail_text}")


# --- 3. Entry Point ---
if __name__ == "__main__":
    app = StreamlitHypothesisApp()
    app.run()
