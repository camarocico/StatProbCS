import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass
from typing import List

# --- 1. Data Models & Logic Layer ---


@dataclass
class AnovaSummaryParams:
    """Data Transfer Object for ANOVA Summary Statistics."""

    means: List[float]  # [Mean1, Mean2, Mean3]
    stds: List[float]  # [SD1, SD2, SD3]
    ns: List[int]  # [n1, n2, n3]
    labels: List[str]  # ["LazySort", "ScareSort", "BribeSort"]
    alpha: float  # Significance Level


class AnovaSummaryCalculator:
    """
    Performs One-Way ANOVA using Summary Statistics (Mean, SD, n).
    """

    def __init__(self, params: AnovaSummaryParams):
        self.p = params
        self.k = len(params.means)  # Number of groups
        self.N = sum(params.ns)  # Total sample size

        self.df_between = self.k - 1
        self.df_within = self.N - self.k

        self.f_stat, self.p_value = self._calculate_anova()
        self.f_critical = self._calculate_critical_val()

    def _calculate_anova(self):
        means = np.array(self.p.means)
        stds = np.array(self.p.stds)
        ns = np.array(self.p.ns)

        # 1. Grand Mean (weighted average of group means)
        grand_mean = np.sum(ns * means) / self.N

        # 2. Sum of Squares Between (SSB)
        # SSB = sum( n_i * (mean_i - grand_mean)^2 )
        ss_between = np.sum(ns * (means - grand_mean) ** 2)

        # 3. Sum of Squares Within (SSW)
        # SSW = sum( (n_i - 1) * variance_i )
        variances = stds**2
        ss_within = np.sum((ns - 1) * variances)

        # 4. Mean Squares
        ms_between = ss_between / self.df_between
        ms_within = ss_within / self.df_within

        # 5. F-Statistic
        if ms_within == 0:
            return 0.0, 1.0  # Edge case: zero variance

        f_stat = ms_between / ms_within

        # 6. P-Value (Right-tailed test)
        p_val = 1 - stats.f.cdf(f_stat, self.df_between, self.df_within)

        return f_stat, p_val

    def _calculate_critical_val(self) -> float:
        return stats.f.ppf(1 - self.p.alpha, self.df_between, self.df_within)

    def is_rejected(self) -> bool:
        return self.f_stat > self.f_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    def __init__(self):
        st.set_page_config(
            page_title="Sorting Algorithm Benchmark", layout="centered", page_icon="⏱️"
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
        calculator = AnovaSummaryCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("⏱️ Algorithm Efficiency Comparison")
        st.markdown("""
        **Scenario:** Comparing the execution times of 3 sorting algorithms using Summary Statistics.
        * **Test:** One-Way ANOVA.
        * **Null Hypothesis ($H_0$):** $\mu_{Lazy} = \mu_{Scare} = \mu_{Bribe}$.
        """)
        st.markdown("---")

    def _render_sidebar(self) -> AnovaSummaryParams:
        st.sidebar.header("1. LazySort Stats")
        m1 = st.sidebar.number_input("Mean (ms)", value=5.21, key="m1")
        s1 = st.sidebar.number_input("Std Dev", value=0.10, min_value=0.01, key="s1")
        n1 = st.sidebar.number_input("Sample Size (n)", value=10, min_value=2, key="n1")

        st.sidebar.markdown("---")
        st.sidebar.header("2. ScareSort Stats")
        m2 = st.sidebar.number_input("Mean (ms)", value=5.43, key="m2")
        s2 = st.sidebar.number_input("Std Dev", value=0.12, min_value=0.01, key="s2")
        n2 = st.sidebar.number_input("Sample Size (n)", value=10, min_value=2, key="n2")

        st.sidebar.markdown("---")
        st.sidebar.header("3. BribeSort Stats")
        m3 = st.sidebar.number_input("Mean (ms)", value=5.51, key="m3")
        s3 = st.sidebar.number_input("Std Dev", value=0.11, min_value=0.01, key="s3")
        n3 = st.sidebar.number_input("Sample Size (n)", value=10, min_value=2, key="n3")

        st.sidebar.markdown("---")
        st.sidebar.header("4. Settings")
        alpha = st.sidebar.selectbox(
            "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
        )

        return AnovaSummaryParams(
            means=[m1, m2, m3],
            stds=[s1, s2, s3],
            ns=[int(n1), int(n2), int(n3)],
            labels=["LazySort", "ScareSort", "BribeSort"],
            alpha=float(alpha),
        )

    def _render_visualization(self, calc: AnovaSummaryCalculator):
        """
        Plots the F-Distribution (Right-skewed).
        """
        dfn, dfd = calc.df_between, calc.df_within
        crit = calc.f_critical
        stat = calc.f_stat

        # Define x range
        # Ensure we cover the statistic if it's large
        max_x = max(crit * 1.5, stat * 1.2, 8.0)
        x = np.linspace(0.01, max_x, 1000)
        y = stats.f.pdf(x, dfn, dfd)

        fig = go.Figure()

        # Regions
        # Acceptance: 0 to Critical
        x_acc = np.linspace(0.01, crit, 200)
        y_acc = stats.f.pdf(x_acc, dfn, dfd)

        # Rejection: Critical to Max
        x_rej = np.linspace(crit, max_x, 200)
        y_rej = stats.f.pdf(x_rej, dfn, dfd)

        # 1. Acceptance Region
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

        # 2. Rejection Region
        fig.add_trace(
            go.Scatter(
                x=x_rej,
                y=y_rej,
                fill="tozeroy",
                fillcolor=self.colors["rejection_fill"],
                name="Rejection Region",
                mode="lines",
                line=dict(color=self.colors["rejection_line"], width=2),
            )
        )

        # 3. Distribution Line
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

        # 4. Observed Statistic
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
                text=f"One-Way ANOVA Result (Confidence = {1 - calc.p.alpha:.2f})",
                font=dict(size=18),
            ),
            xaxis_title=dict(text="F Statistic", font=dict(size=16)),
            yaxis_title=dict(text="Density", font=dict(size=16)),
            legend=dict(
                yanchor="top", y=0.99, xanchor="right", x=0.79, font=dict(size=14)
            ),
            template="plotly_white",
            height=600,
            margin=dict(t=50, b=50, l=50, r=50),
        )

        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "anova_summary_result",
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

    def _render_results(self, calc: AnovaSummaryCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)

        col1.metric("F-Statistic", f"{calc.f_stat:.3f}")
        col2.metric("Critical F", f"> {calc.f_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Algorithms differ significantly."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "No significant difference found."
            delta_color = "normal"

        col3.metric(
            "Conclusion",
            decision_text,
            f"p = {calc.p_value:.6f}",
            delta_color=delta_color,
        )

        if calc.is_rejected():
            st.error(
                f"**Action Required:** {detail_text} One algorithm is performing differently."
            )
        else:
            st.success(
                f"**Status:** {detail_text} Performance is statistically identical."
            )


# --- 3. Entry Point ---
if __name__ == "__main__":
    app = StreamlitHypothesisApp()
    app.run()
