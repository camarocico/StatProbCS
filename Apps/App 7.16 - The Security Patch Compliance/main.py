import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass

# --- 1. Data Models & Logic Layer ---


@dataclass
class ProportionTestParameters:
    """Data Transfer Object for proportion test parameters."""

    p_0: float  # Hypothesized Proportion
    sample_successes: int  # Number of "successes" (x)
    n: int  # Sample Size
    alpha: float  # Significance Level


class ProportionZTestCalculator:
    """
    Handles the statistical logic for a Left-Tailed Z-Test for Proportions.
    (Testing if actual p < claimed p)
    """

    def __init__(self, params: ProportionTestParameters):
        self.params = params
        self.p_hat = self.params.sample_successes / self.params.n
        self.se = self._calculate_standard_error()
        self.z_score = self._calculate_z_score()
        self.z_critical = self._calculate_critical_z()

    def _calculate_standard_error(self) -> float:
        # Standard Error uses p_0 (under Null Hypothesis)
        p0 = self.params.p_0
        return np.sqrt((p0 * (1 - p0)) / self.params.n)

    def _calculate_z_score(self) -> float:
        return (self.p_hat - self.params.p_0) / self.se

    def _calculate_critical_z(self) -> float:
        # One-tailed (Left) critical value
        # We want the Z score that cuts off the bottom alpha
        return stats.norm.ppf(self.params.alpha)

    def get_p_value(self) -> float:
        # Area to the left of the Z-score
        return stats.norm.cdf(self.z_score)

    def get_critical_cutoff_value(self) -> float:
        """Returns the raw proportion value where the rejection region starts."""
        # Margin of error (negative because left tail)
        margin = self.z_critical * self.se
        return self.params.p_0 + margin

    def is_rejected(self) -> bool:
        # Reject if observed Z is less (more negative) than critical Z
        return self.z_score < self.z_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    def __init__(self):
        st.set_page_config(
            page_title="Security Patch Compliance Test",
            layout="centered",
            page_icon="🛡️",
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
        calculator = ProportionZTestCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("🛡️ Security Compliance Hypothesis Test")
        st.markdown("""
        **Scenario:** The CISO claims **at least 80%** ($p_0$) of devices are patched. 
        You suspect the rate is **lower**.
        * **Null Hypothesis ($H_0$):** $p \ge 0.80$
        * **Alternative ($H_1$):** $p < 0.80$ (Left-Tailed)
        """)
        st.markdown("---")

    def _render_sidebar(self) -> ProportionTestParameters:
        st.sidebar.header("1. CISO's Claim ($H_0$)")
        p_0 = st.sidebar.slider("Claimed Proportion ($p_0$)", 0.5, 1.0, 0.8, 0.05)

        st.sidebar.header("2. Audit Data")
        n = st.sidebar.number_input("Sample Size (n)", value=500, step=10)

        # Helper to set initial successes based on typical problem values
        default_successes = int(n * 0.75)  # 375 for n=500
        sample_successes = st.sidebar.slider(
            "Patched Devices Found (x)",
            min_value=0,
            max_value=n,
            value=default_successes,
            step=1,
        )

        current_p_hat = sample_successes / n
        st.sidebar.caption(f"Sample Proportion ($\hat{{p}}$): **{current_p_hat:.2%}**")

        st.sidebar.header("3. Risk Tolerance")
        alpha = st.sidebar.selectbox(
            "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
        )

        return ProportionTestParameters(p_0, sample_successes, int(n), float(alpha))

    def _render_visualization(self, calc: ProportionZTestCalculator):
        """
        Plots the Normal Approximation of the Sampling Distribution (Left-Tailed).
        """
        p0 = calc.params.p_0
        se = calc.se

        # Cutoff in terms of Proportion (not Z)
        x_crit = calc.get_critical_cutoff_value()

        # Define x range (Proportion axis)
        # We plot +/- 4 standard errors from the null proportion
        x = np.linspace(p0 - 4 * se, p0 + 4 * se, 1000)
        y = stats.norm.pdf(x, p0, se)

        # Define specific regions for shading
        # 1. Rejection Region (Left of Critical Value)
        x_fill_reject = np.linspace(x[0], x_crit, 200)
        y_fill_reject = stats.norm.pdf(x_fill_reject, p0, se)

        # 2. Acceptance Region (Right of Critical Value)
        x_fill_accept = np.linspace(x_crit, x[-1], 200)
        y_fill_accept = stats.norm.pdf(x_fill_accept, p0, se)

        fig = go.Figure()

        # 1. Rejection Region (Salmon)
        fig.add_trace(
            go.Scatter(
                x=x_fill_reject,
                y=y_fill_reject,
                fill="tozeroy",
                fillcolor=self.colors["rejection_fill"],
                name="Left Rejection Region",
                mode="lines",
                line=dict(color=self.colors["rejection_line"], width=2),
            )
        )

        # 2. Acceptance Region (Dark Cyan)
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

        # 3. Distribution Line (Blue)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name="Sampling Distribution ($H_0$)",
                mode="lines",
                opacity=0.8,
                line=dict(color=self.colors["dist_line"], width=2),
            )
        )

        # 4. Statistic (Purple)
        fig.add_vline(
            x=calc.p_hat,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"Observed $\hat{{p}}$ = {calc.p_hat:.3f}",
            annotation_position="top left",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        fig.update_layout(
            title=dict(
                text=f"Left-Tailed Proportion Test (Confidence = {1 - calc.params.alpha:.2f})",
                font=dict(size=18),
            ),
            xaxis_title=dict(text="Proportion of Patched Devices", font=dict(size=16)),
            yaxis_title=dict(text="Density", font=dict(size=16)),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.65, font=dict(size=14)
            ),
            template="plotly_white",
            height=600,
            margin=dict(t=50, b=50, l=50, r=50),
        )

        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "security_audit_result",
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

    def _render_results(self, calc: ProportionZTestCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)

        col1.metric("Z-Score", f"{calc.z_score:.3f}")
        col2.metric("Critical Z Threshold", f"< {calc.z_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Compliance is significantly lower than 80%."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "Compliance is not significantly lower than claimed."
            delta_color = "normal"

        col3.metric(
            "Conclusion",
            decision_text,
            f"p = {calc.get_p_value():.4f}",
            delta_color=delta_color,
        )

        if calc.is_rejected():
            st.error(f"**Action Required:** {detail_text} Initiate remediation.")
        else:
            st.success(f"**Status:** {detail_text} No immediate action required.")


# --- 3. Entry Point ---
if __name__ == "__main__":
    app = StreamlitHypothesisApp()
    app.run()
