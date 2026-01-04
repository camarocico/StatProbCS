import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass

# --- 1. Data Models & Logic Layer ---


@dataclass
class TestParameters:
    """Data Transfer Object for test parameters."""

    mu_0: float  # Population Mean (Hypothesized)
    sample_mean: float  # Observed Sample Mean
    sigma: float  # Population Standard Deviation
    n: int  # Sample Size
    alpha: float  # Significance Level


class OneTailedZTestCalculator:
    """
    Handles the statistical logic for a RIGHT-TAILED Z-test.
    (Testing if actual mean > claimed mean)
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
        # One-tailed (Right) critical value
        # We want the Z score that cuts off the top alpha%
        return stats.norm.ppf(1 - self.params.alpha)

    def get_p_value(self) -> float:
        # Area to the right of the Z-score
        return 1 - stats.norm.cdf(self.z_score)

    def get_critical_cutoff_value(self) -> float:
        """Returns the raw X value where the rejection region starts."""
        margin_of_error = self.z_critical * self.se
        return self.params.mu_0 + margin_of_error

    def is_rejected(self) -> bool:
        # Reject if observed Z is greater than critical Z
        return self.z_score > self.z_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    """
    Manages the Streamlit User Interface.
    """

    def __init__(self):
        st.set_page_config(
            page_title="Compilation Time Test", layout="centered", page_icon="⏱️"
        )
        # Style matching your reference
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
        calculator = OneTailedZTestCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("⏱️ Compilation Time Hypothesis Test")
        st.markdown("""
        **Scenario:** A company claims their new software compiles code in **at most 10 minutes**.
        You suspect it takes **longer**.
        * **Null Hypothesis ($H_0$):** $\mu \le 10$ min.
        * **Alternative ($H_1$):** $\mu > 10$ min (Right-Tailed).
        """)
        st.markdown("---")

    def _render_sidebar(self) -> TestParameters:
        st.sidebar.header("1. Configuration")
        with st.sidebar.expander("Claim Definitions ($H_0$)", expanded=True):
            mu_0 = st.number_input("Claimed Mean Time (min)", value=10.0, step=0.1)
            n = st.number_input("Sample Size (n)", value=50, step=1)
            alpha = st.selectbox(
                "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
            )

        st.sidebar.header("2. Observations")
        # Default variance is 4, so sigma is 2
        initial_var = 4.0
        sigma_sq = st.sidebar.slider(
            "Population Variance ($\sigma^2$)",
            min_value=0.1,
            max_value=20.0,
            value=initial_var,
            step=0.1,
        )
        sigma = np.sqrt(sigma_sq)
        st.sidebar.caption(f"Implied Std Dev ($\sigma$): **{sigma:.2f} min**")

        sample_mean = st.sidebar.slider(
            "Observed Sample Mean (min)",
            min_value=8.0,
            max_value=12.0,
            value=10.5,
            step=0.1,
        )

        return TestParameters(mu_0, sample_mean, sigma, int(n), float(alpha))

    def _render_visualization(self, calc: OneTailedZTestCalculator):
        """
        Plots the distribution for a Right-Tailed Test.
        """
        mu = calc.params.mu_0
        se = calc.se
        x_crit = calc.get_critical_cutoff_value()

        # Define x range for the plot
        x = np.linspace(mu - 4 * se, mu + 4 * se, 1000)
        y = stats.norm.pdf(x, mu, se)

        # Define specific regions for shading
        # 1. Non-Rejection Region (Left of Critical Value)
        x_fill_accept = np.linspace(x[0], x_crit, 200)
        y_fill_accept = stats.norm.pdf(x_fill_accept, mu, se)

        # 2. Rejection Region (Right of Critical Value)
        x_fill_reject = np.linspace(x_crit, x[-1], 200)
        y_fill_reject = stats.norm.pdf(x_fill_reject, mu, se)

        fig = go.Figure()

        # 1. Acceptance Region (Dark Cyan)
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

        # 2. Rejection Region (Salmon)
        fig.add_trace(
            go.Scatter(
                x=x_fill_reject,
                y=y_fill_reject,
                fill="tozeroy",
                fillcolor=self.colors["rejection_fill"],
                name="Right Rejection Region",
                mode="lines",
                line=dict(color=self.colors["rejection_line"], width=2),
            )
        )

        # 3. The Standard Normal Line (Blue)
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

        # 4. The Observed Statistic (Purple Dashed Line)
        fig.add_vline(
            x=calc.params.sample_mean,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"Observed Mean = {calc.params.sample_mean:.2f}",
            annotation_position="top right",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        # Layout styling
        fig.update_layout(
            title=dict(
                text=f"Right-Sided Test (Confidence level = {1 - calc.params.alpha:.2f})",
                font=dict(size=18),
            ),
            xaxis_title=dict(text="Compilation Time (min)", font=dict(size=16)),
            yaxis_title=dict(text="Density", font=dict(size=16)),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=14)
            ),
            template="plotly_white",
            height=600,
            margin=dict(t=50, b=50, l=50, r=50),
        )

        # High-Res Download Config
        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "compilation_test_result",
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

    def _render_results(self, calc: OneTailedZTestCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)

        # Determine Critical Z display string (one-sided)
        col1.metric("Z-Score", f"{calc.z_score:.3f}")
        col2.metric("Critical Z Threshold", f"> {calc.z_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Evidence suggests average time > 10 mins."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "Not enough evidence to disprove the claim."
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
