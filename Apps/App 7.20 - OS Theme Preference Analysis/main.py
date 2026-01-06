from dataclasses import dataclass
from typing import List

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# --- 1. Data Models & Logic Layer ---


@dataclass
class ChiSquareParameters:
    """Data Transfer Object for Chi-Square test."""

    observed: List[int]  # Observed counts [Light, Dark, Classic]
    expected_probs: List[float]  # Expected proportions [0.5, 0.3, 0.2]
    alpha: float  # Significance Level
    categories: List[str]  # Names of categories


class ChiSquareCalculator:
    """
    Handles statistical logic for Chi-Square Goodness of Fit.
    """

    def __init__(self, params: ChiSquareParameters):
        self.p = params
        self.n = sum(self.p.observed)
        self.k = len(self.p.observed)
        self.df = self.k - 1

        self.expected_counts = [prob * self.n for prob in self.p.expected_probs]
        self.chi2_stat = self._calculate_chi2_stat()
        self.chi2_critical = self._calculate_critical_val()

    def _calculate_chi2_stat(self) -> float:
        chi2 = 0.0
        for obs, exp in zip(self.p.observed, self.expected_counts):
            if exp > 0:
                chi2 += ((obs - exp) ** 2) / exp
        return chi2

    def _calculate_critical_val(self) -> float:
        # Chi-Square is always right-tailed test for Goodness of Fit
        return stats.chi2.ppf(1 - self.p.alpha, self.df)

    def get_p_value(self) -> float:
        return 1 - stats.chi2.cdf(self.chi2_stat, self.df)

    def is_rejected(self) -> bool:
        return self.chi2_stat > self.chi2_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    def __init__(self):
        st.set_page_config(
            page_title="UX Theme Preference Test", layout="centered", page_icon="🎨"
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

        # Validation: Check if sums match
        if abs(sum(params.expected_probs) - 1.0) > 0.01:
            st.error("Error: Hypothesized proportions must sum to 1.0")
            return

        calculator = ChiSquareCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("🎨 OS Theme Preference Analysis")
        st.markdown("""
        **Scenario:** Testing if user preferences for Light/Dark/Classic themes match the company's claim.
        * **Test:** Chi-Square Goodness of Fit ($\chi^2$).
        * **Null Hypothesis ($H_0$):** Preferences follow the claimed distribution.
        """)
        st.markdown("---")

    def _render_sidebar(self) -> ChiSquareParameters:
        st.sidebar.header("1. Claimed Proportions ($H_0$)")
        # Fixed for this specific problem, or could be made editable
        p_light = 0.5
        p_dark = 0.3
        p_classic = 0.2

        # Display as disabled inputs or just text
        st.sidebar.markdown(
            f"**Light:** {p_light}, **Dark:** {p_dark}, **Classic:** {p_classic}"
        )

        st.sidebar.markdown("---")
        st.sidebar.header("2. Survey Data (Observed)")

        obs_light = st.sidebar.number_input("Light Theme Users", value=240, step=1)
        obs_dark = st.sidebar.number_input("Dark Theme Users", value=155, step=1)
        obs_classic = st.sidebar.number_input("Classic Theme Users", value=105, step=1)

        n = obs_light + obs_dark + obs_classic
        st.sidebar.caption(f"Total Sample Size ($n$): {n}")

        st.sidebar.markdown("---")
        st.sidebar.header("3. Settings")
        alpha = st.sidebar.selectbox(
            "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
        )

        return ChiSquareParameters(
            observed=[obs_light, obs_dark, obs_classic],
            expected_probs=[p_light, p_dark, p_classic],
            alpha=float(alpha),
            categories=["Light", "Dark", "Classic"],
        )

    def _render_visualization(self, calc: ChiSquareCalculator):
        """
        Plots the Chi-Square Distribution.
        """
        df = calc.df
        crit = calc.chi2_critical

        # Define x range for Chi-Square (starts at 0)
        # Go far enough to show the tail or the statistic
        max_x = max(crit * 1.5, calc.chi2_stat * 1.2, 10.0)
        x = np.linspace(0, max_x, 1000)
        y = stats.chi2.pdf(x, df)

        # Regions
        # Acceptance: 0 to critical value
        x_accept = np.linspace(0, crit, 200)
        y_accept = stats.chi2.pdf(x_accept, df)

        # Rejection: critical value onwards
        x_reject = np.linspace(crit, max_x, 200)
        y_reject = stats.chi2.pdf(x_reject, df)

        fig = go.Figure()

        # 1. Acceptance Region
        fig.add_trace(
            go.Scatter(
                x=x_accept,
                y=y_accept,
                fill="tozeroy",
                fillcolor=self.colors["accept_fill"],
                name="Non-Rejection Region",
                mode="lines",
                line=dict(color=self.colors["accept_line"], width=2),
            )
        )

        # 2. Rejection Region (Right Tail)
        fig.add_trace(
            go.Scatter(
                x=x_reject,
                y=y_reject,
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
                name=f"Chi-Square Dist (df={df})",
                mode="lines",
                opacity=0.8,
                line=dict(color=self.colors["dist_line"], width=2),
            )
        )

        # 4. Observed Statistic
        fig.add_vline(
            x=calc.chi2_stat,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"Observed $\chi^2$: {calc.chi2_stat:.2f}",
            annotation_position="top right",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        fig.update_layout(
            title=dict(
                text=f"Chi-Square Goodness of Fit (Confidence = {1 - calc.p.alpha:.2f})",
                font=dict(size=18),
            ),
            xaxis_title=dict(text="$\chi^2$ Statistic", font=dict(size=16)),
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
                "filename": "chi_square_result",
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

    def _render_results(self, calc: ChiSquareCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)

        col1.metric("Chi-Square Stat", f"{calc.chi2_stat:.3f}")
        col2.metric("Critical Value", f"> {calc.chi2_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Observed preferences differ significantly from the claim."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "Observed preferences match the claim reasonably well."
            delta_color = "normal"

        col3.metric(
            "Conclusion",
            decision_text,
            f"p = {calc.get_p_value():.4f}",
            delta_color=delta_color,
        )

        # Comparison Table
        st.markdown("### Data Comparison")
        comp_data = {
            "Category": calc.p.categories,
            "Observed": calc.p.observed,
            "Expected": [f"{x:.1f}" for x in calc.expected_counts],
            "Contribution to $\chi^2$": [
                f"{((o - e) ** 2) / e:.3f}"
                for o, e in zip(calc.p.observed, calc.expected_counts)
            ],
        }
        st.table(comp_data)

        if calc.is_rejected():
            st.error(
                f"**Action Required:** {detail_text} The UX design assumptions may need revisiting."
            )
        else:
            st.success(f"**Status:** {detail_text} The design assumptions are valid.")


# --- 3. Entry Point ---
if __name__ == "__main__":
    app = StreamlitHypothesisApp()
    app.run()
