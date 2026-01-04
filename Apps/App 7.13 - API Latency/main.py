import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass
from typing import Tuple

# --- 1. Data Models & Logic Layer (Unchanged) ---


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
            page_title="DevOps Hypothesis Test", layout="centered", page_icon="🖥️"
        )
        self.colors = {
            "h0": "#2c3e50",  # Dark Blue
            "reject": "#e74c3c",  # Red
            "accept": "#27ae60",  # Green
            "obs_line": "#2980b9",  # Blue
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
                "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=0
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
        """Plots the Sampling Distribution using Plotly."""

        mu = calc.params.mu_0
        se = calc.se
        x_left_crit, x_right_crit = calc.get_critical_region_bounds()

        # Define x range
        x = np.linspace(mu - 4 * se, mu + 4 * se, 1000)
        y = stats.norm.pdf(x, mu, se)

        fig = go.Figure()

        # 1. Main Null Distribution Curve
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Null Distribution",
                line=dict(color=self.colors["h0"], width=2),
                hoverinfo="x+y",
            )
        )

        # 2. Right Rejection Region (Shaded Area)
        x_right = np.linspace(x_right_crit, mu + 4 * se, 200)
        y_right = stats.norm.pdf(x_right, mu, se)
        # Add a zero at the end to close the polygon nicely for filling
        x_right_poly = np.concatenate([x_right, [x_right[-1], x_right[0]]])
        y_right_poly = np.concatenate([y_right, [0, 0]])

        fig.add_trace(
            go.Scatter(
                x=x_right,
                y=y_right,
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(231, 76, 60, 0.3)",  # Red with opacity
                name="Rejection Region (> Critical)",
                hoverinfo="skip",
            )
        )

        # 3. Left Rejection Region (Shaded Area)
        x_left = np.linspace(mu - 4 * se, x_left_crit, 200)
        y_left = stats.norm.pdf(x_left, mu, se)

        fig.add_trace(
            go.Scatter(
                x=x_left,
                y=y_left,
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(231, 76, 60, 0.3)",
                name="Rejection Region (< Critical)",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # 4. Vertical Line for Sample Mean
        obs_color = (
            self.colors["reject"] if calc.is_rejected() else self.colors["accept"]
        )

        fig.add_vline(
            x=calc.params.sample_mean,
            line_width=3,
            line_dash="dash",
            line_color=obs_color,
            annotation_text=f"Observed: {calc.params.sample_mean:.1f}",
            annotation_position="top right",
        )

        # Layout styling
        fig.update_layout(
            title="Sampling Distribution of the Mean",
            xaxis_title="Latency (ms)",
            yaxis_title="Probability Density",
            template="plotly_white",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Plotly chart with built-in download button enabled in the modebar
        st.plotly_chart(fig, use_container_width=True)

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
