import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass

# --- 1. Data Models & Logic Layer ---


@dataclass
class FTestParameters:
    """Data Transfer Object for F-Test parameters."""

    s1: float
    n1: int  # Group 1 (Numerator Candidates)
    s2: float
    n2: int  # Group 2
    alpha: float  # Significance Level


class FTestCalculator:
    """
    Handles statistical logic for F-Test of Equality of Variances.
    Automatically places the larger variance in the numerator for the statistic,
    but handles two-tailed logic correctly.
    """

    def __init__(self, params: FTestParameters):
        self.p = params

        # Determine which is larger to follow F-test convention (F > 1)
        # This makes visualization of the 'upper' tail easier to understand
        if self.p.s1**2 >= self.p.s2**2:
            self.var_num = self.p.s1**2
            self.var_den = self.p.s2**2
            self.df_num = self.p.n1 - 1
            self.df_den = self.p.n2 - 1
            self.num_label = "Group 1"
        else:
            self.var_num = self.p.s2**2
            self.var_den = self.p.s1**2
            self.df_num = self.p.n2 - 1
            self.df_den = self.p.n1 - 1
            self.num_label = "Group 2"

        self.f_stat = self.var_num / self.var_den

        # Critical Values (Two-Tailed split)
        # We calculate the upper critical value for the rejection check
        self.f_crit_upper = stats.f.ppf(1 - self.p.alpha / 2, self.df_num, self.df_den)
        self.f_crit_lower = stats.f.ppf(self.p.alpha / 2, self.df_num, self.df_den)

    def get_p_value(self) -> float:
        # P-value for two-tailed test
        # Area to the right of F_stat * 2
        p_one_tail = 1 - stats.f.cdf(self.f_stat, self.df_num, self.df_den)
        return min(p_one_tail * 2, 1.0)  # Cap at 1.0

    def is_rejected(self) -> bool:
        return self.f_stat > self.f_crit_upper


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    def __init__(self):
        st.set_page_config(
            page_title="Variance Comparison (F-Test)", layout="centered", page_icon="📊"
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
        calculator = FTestCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("📊 Grade Variance Comparison")
        st.markdown("""
        **Scenario:** Comparing the spread (variance) of grades between two courses.
        * **Test:** F-Test for Equality of Variances.
        * **Null Hypothesis ($H_0$):** $\sigma_1^2 = \sigma_2^2$ (Equal spread).
        """)
        st.markdown("---")

    def _render_sidebar(self) -> FTestParameters:
        st.sidebar.header("1. Course A (e.g., Art)")
        n1 = st.sidebar.number_input("Sample Size ($n_A$)", value=30, step=1)
        s1 = st.sidebar.number_input("Std Deviation ($s_A$)", value=3.5, step=0.1)

        st.sidebar.markdown("---")

        st.sidebar.header("2. Course B (e.g., CS)")
        n2 = st.sidebar.number_input("Sample Size ($n_B$)", value=25, step=1)
        s2 = st.sidebar.number_input("Std Deviation ($s_B$)", value=5.7, step=0.1)

        st.sidebar.markdown("---")
        st.sidebar.header("3. Settings")
        alpha = st.sidebar.selectbox(
            "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
        )

        return FTestParameters(float(s1), int(n1), float(s2), int(n2), float(alpha))

    def _render_visualization(self, calc: FTestCalculator):
        """
        Plots the F-Distribution.
        """
        dfn, dfd = calc.df_num, calc.df_den
        crit_upper = calc.f_crit_upper
        crit_lower = calc.f_crit_lower
        stat = calc.f_stat

        # Define x range (F-dist starts at 0)
        # Go far enough to show statistic or upper critical value
        max_x = max(crit_upper * 1.5, stat * 1.2, 5.0)
        x = np.linspace(
            0.01, max_x, 1000
        )  # Start slightly > 0 to avoid div/0 in PDF sometimes
        y = stats.f.pdf(x, dfn, dfd)

        fig = go.Figure()

        # Regions definition
        # Left Rejection (0 to Lower Crit)
        x_rej_left = np.linspace(0.01, crit_lower, 100)
        y_rej_left = stats.f.pdf(x_rej_left, dfn, dfd)

        # Acceptance (Lower Crit to Upper Crit)
        x_acc = np.linspace(crit_lower, crit_upper, 200)
        y_acc = stats.f.pdf(x_acc, dfn, dfd)

        # Right Rejection (Upper Crit onwards)
        x_rej_right = np.linspace(crit_upper, max_x, 200)
        y_rej_right = stats.f.pdf(x_rej_right, dfn, dfd)

        # 1. Left Rejection Region
        fig.add_trace(
            go.Scatter(
                x=x_rej_left,
                y=y_rej_left,
                fill="tozeroy",
                fillcolor=self.colors["rejection_fill"],
                name="Rejection Region",
                mode="lines",
                line=dict(color=self.colors["rejection_line"], width=2),
            )
        )

        # 2. Acceptance Region
        fig.add_trace(
            go.Scatter(
                x=x_acc,
                y=y_acc,
                fill="tozeroy",
                fillcolor=self.colors["accept_fill"],
                name="Non-Rejection Region",
                mode="lines",
                line=dict(color=self.colors["accept_line"], width=2),
            )
        )

        # 3. Right Rejection Region
        fig.add_trace(
            go.Scatter(
                x=x_rej_right,
                y=y_rej_right,
                fill="tozeroy",
                fillcolor=self.colors["rejection_fill"],
                name="Rejection Region",
                showlegend=False,
                mode="lines",
                line=dict(color=self.colors["rejection_line"], width=2),
            )
        )

        # 4. Distribution Line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=f"F-Dist ($df_1={dfn}, df_2={dfd}$)",
                mode="lines",
                opacity=0.8,
                line=dict(color=self.colors["dist_line"], width=2),
            )
        )

        # 5. Observed Statistic
        fig.add_vline(
            x=stat,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"F-Stat: {stat:.2f}",
            annotation_position="top right",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        fig.update_layout(
            title=dict(
                text=f"F-Distribution (Confidence = {1 - calc.p.alpha:.2f})",
                font=dict(size=18),
            ),
            xaxis_title=dict(text="F Statistic", font=dict(size=16)),
            yaxis_title=dict(text="Density", font=dict(size=16)),
            legend=dict(
                yanchor="top", y=0.99, xanchor="right", x=0.99, font=dict(size=14)
            ),
            template="plotly_white",
            height=600,
            margin=dict(t=50, b=50, l=50, r=50),
        )

        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "ftest_variance_result",
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

    def _render_results(self, calc: FTestCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)

        col1.metric("F-Statistic", f"{calc.f_stat:.3f}")
        col2.metric("Upper Critical F", f"> {calc.f_crit_upper:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Variances are significantly different."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "Variances are consistent with being equal."
            delta_color = "normal"

        col3.metric(
            "Conclusion",
            decision_text,
            f"p = {calc.get_p_value():.4f}",
            delta_color=delta_color,
        )

        st.caption(
            f"Note: The statistic was calculated using **{calc.num_label}** in the numerator (Larger Variance)."
        )

        if calc.is_rejected():
            st.error(
                f"**Action Required:** {detail_text} The Computer Science course likely has more variability in grades."
            )
        else:
            st.success(
                f"**Status:** {detail_text} No significant evidence that one course varies more than the other."
            )


# --- 3. Entry Point ---
if __name__ == "__main__":
    app = StreamlitHypothesisApp()
    app.run()
