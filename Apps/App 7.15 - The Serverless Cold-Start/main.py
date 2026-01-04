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

    mu_0: float  # Hypothesized Mean
    sample_mean: float  # Observed Sample Mean
    s: float  # Sample Standard Deviation (s)
    n: int  # Sample Size
    alpha: float  # Significance Level


class TTestCalculator:
    """
    Handles the statistical logic for a Two-Tailed T-Test.
    Uses Student's t-distribution (df = n - 1).
    """

    def __init__(self, params: TestParameters):
        self.params = params
        self.df = params.n - 1  # Degrees of Freedom
        self.se = self._calculate_standard_error()
        self.t_score = self._calculate_t_score()
        self.t_critical = self._calculate_critical_t()

    def _calculate_standard_error(self) -> float:
        return self.params.s / np.sqrt(self.params.n)

    def _calculate_t_score(self) -> float:
        return (self.params.sample_mean - self.params.mu_0) / self.se

    def _calculate_critical_t(self) -> float:
        # Two-tailed critical value for t-distribution
        return stats.t.ppf(1 - self.params.alpha / 2, self.df)

    def get_p_value(self) -> float:
        return 2 * (1 - stats.t.cdf(abs(self.t_score), self.df))

    def get_critical_region_bounds(self) -> Tuple[float, float]:
        """Returns the raw X values where rejection regions start."""
        margin_of_error = self.t_critical * self.se
        return (self.params.mu_0 - margin_of_error, self.params.mu_0 + margin_of_error)

    def is_rejected(self) -> bool:
        return abs(self.t_score) > self.t_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    def __init__(self):
        st.set_page_config(
            page_title="Serverless Latency T-Test", layout="centered", page_icon="☁️"
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
        calculator = TTestCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("☁️ Serverless Latency T-Test")
        st.markdown("""
        **Scenario:** A provider claims **100 ms** average cold-start time. 
        You test a sample of $n$ functions and calculate the sample standard deviation ($s$).
        * **Null Hypothesis ($H_0$):** $\mu = 100$ ms.
        * **Alternative ($H_1$):** $\mu \\neq 100$ ms.
        
        *Note: Since we use sample standard deviation ($s$), this is a **T-Test**.*
        """)
        st.markdown("---")

    def _render_sidebar(self) -> TestParameters:
        st.sidebar.header("1. Configuration")
        with st.sidebar.expander("Claim Definitions ($H_0$)", expanded=True):
            mu_0 = st.number_input("Claimed Latency (ms)", value=100.0, step=0.1)
            n = st.number_input("Sample Size (n)", value=30, step=1)
            alpha = st.selectbox(
                "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
            )

        st.sidebar.header("2. Field Data")
        # User enters sample SD directly
        s = st.sidebar.slider(
            "Sample Std Dev (s)", min_value=1.0, max_value=20.0, value=8.0, step=0.1
        )
        st.sidebar.caption(f"Variance ($s^2$): **{s**2:.2f}**")

        sample_mean = st.sidebar.slider(
            "Observed Sample Mean (ms)",
            min_value=90.0,
            max_value=110.0,
            value=102.0,
            step=0.1,
        )

        return TestParameters(mu_0, sample_mean, s, int(n), float(alpha))

    def _render_visualization(self, calc: TTestCalculator):
        """
        Plots the Student's t-distribution.
        """
        mu = calc.params.mu_0
        se = calc.se
        df = calc.df
        x_left_crit, x_right_crit = calc.get_critical_region_bounds()

        # Define x range (using standard errors)
        x = np.linspace(mu - 4 * se, mu + 4 * se, 1000)
        # Use stats.t.pdf instead of norm.pdf
        y = stats.t.pdf(x, df, loc=mu, scale=se)

        # Regions
        x_fill_left = np.linspace(x[0], x_left_crit, 100)
        y_fill_left = stats.t.pdf(x_fill_left, df, loc=mu, scale=se)

        x_fill_right = np.linspace(x_right_crit, x[-1], 100)
        y_fill_right = stats.t.pdf(x_fill_right, df, loc=mu, scale=se)

        x_fill_accept = np.linspace(x_left_crit, x_right_crit, 100)
        y_fill_accept = stats.t.pdf(x_fill_accept, df, loc=mu, scale=se)

        fig = go.Figure()

        # 1. Left Rejection (Salmon)
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

        # 2. Right Rejection (Salmon)
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

        # 3. Acceptance (Dark Cyan)
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

        # 4. Distribution Line (Blue)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=f"T-Distribution (df={df})",
                mode="lines",
                opacity=0.8,
                line=dict(color=self.colors["dist_line"], width=2),
            )
        )

        # 5. Statistic (Purple)
        fig.add_vline(
            x=calc.params.sample_mean,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"Observed: {calc.params.sample_mean:.1f}",
            annotation_position="top left",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        fig.update_layout(
            title=dict(
                text=f"T-Test Rejection Regions (Confidence = {1 - calc.params.alpha:.2f})",
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

        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "t_test_result_high_res",
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

    def _render_results(self, calc: TTestCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)
        col1.metric("t-Score", f"{calc.t_score:.3f}")
        col2.metric("Critical t Threshold", f"±{calc.t_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Statistically significant deviation from 100 ms."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "Deviation is likely due to random chance."
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
