import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from dataclasses import dataclass

# --- 1. Data Models & Logic Layer ---


@dataclass
class ChiSquareIndepParameters:
    """Data Transfer Object for Independence Test."""

    observed_df: pd.DataFrame  # The contingency table
    alpha: float  # Significance Level


class ChiSquareIndepCalculator:
    """
    Handles statistical logic for Chi-Square Test of Independence.
    """

    def __init__(self, params: ChiSquareIndepParameters):
        self.p = params
        self.observed = self.p.observed_df.values

        # Calculate Chi2, p-value, dof, and expected frequencies
        self.chi2_stat, self.p_value, self.dof, self.expected = stats.chi2_contingency(
            self.observed
        )

        self.chi2_critical = self._calculate_critical_val()

    def _calculate_critical_val(self) -> float:
        # Chi-Square Independence test is always right-tailed
        return stats.chi2.ppf(1 - self.p.alpha, self.dof)

    def is_rejected(self) -> bool:
        return self.chi2_stat > self.chi2_critical


# --- 2. Presentation Layer (Streamlit + Plotly) ---


class StreamlitHypothesisApp:
    def __init__(self):
        st.set_page_config(
            page_title="Dev Environment Preference Test",
            layout="centered",
            page_icon="🐧",
        )
        # Colors matching your requested style (Fig_07_01.py)
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
        calculator = ChiSquareIndepCalculator(params)
        self._render_visualization(calculator)
        self._render_results(calculator)

    def _render_header(self):
        st.title("🐧 OS vs Language Association Test")
        st.markdown("""
        **Scenario:** Testing if Operating System preference is associated with Programming Language.
        * **Test:** Chi-Square Test for Independence ($\chi^2$).
        * **Null Hypothesis ($H_0$):** Variables are **Independent**.
        * **Alternative ($H_1$):** Variables are **Associated**.
        """)
        st.markdown("---")

    def _render_sidebar(self) -> ChiSquareIndepParameters:
        st.sidebar.header("1. Contingency Table (Observed)")

        # Default Data matching the problem
        data = {"Windows": [50, 40, 60], "MacOS": [30, 35, 25], "Linux": [20, 25, 15]}
        index = ["Python", "Java", "C++"]

        # Create an editable dataframe
        df_input = pd.DataFrame(data, index=index)

        st.sidebar.info("Edit the values in the table below:")
        edited_df = st.sidebar.data_editor(df_input)

        # Calculate totals for display
        total_n = edited_df.values.sum()
        st.sidebar.caption(f"Total Sample Size ($n$): {total_n}")

        st.sidebar.markdown("---")
        st.sidebar.header("2. Settings")
        alpha = st.sidebar.selectbox(
            "Significance Level ($\\alpha$)", [0.01, 0.05, 0.10], index=1
        )

        return ChiSquareIndepParameters(edited_df, float(alpha))

    def _render_visualization(self, calc: ChiSquareIndepCalculator):
        """
        Plots the Chi-Square Distribution using the requested visual style.
        """
        df = calc.dof
        crit = calc.chi2_critical
        stat = calc.chi2_stat

        # Define x range
        max_x = max(crit * 1.5, stat * 1.2, 15.0)
        x = np.linspace(0, max_x, 1000)
        y = stats.chi2.pdf(x, df)

        # Regions
        # Acceptance: 0 to critical value
        x_accept = np.linspace(0, crit, 200)
        y_accept = stats.chi2.pdf(x_accept, df)

        # Rejection: critical value onwards (Right Tail)
        x_reject = np.linspace(crit, max_x, 200)
        y_reject = stats.chi2.pdf(x_reject, df)

        fig = go.Figure()

        # 1. Acceptance Region (Dark Cyan)
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

        # 2. Rejection Region (Salmon - Right Tail)
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

        # 3. Distribution Line (Blue)
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

        # 4. Observed Statistic (Purple Dashed)
        fig.add_vline(
            x=stat,
            line_width=2,
            line_dash="dash",
            line_color=self.colors["statistic"],
            annotation_text=f"Observed $\chi^2$: {stat:.2f}",
            annotation_position="top right",
            annotation_font_size=14,
            annotation_font_color=self.colors["statistic"],
            annotation_textangle=270,
        )

        fig.update_layout(
            title=dict(
                text=f"Chi-Square Independence (Confidence = {1 - calc.p.alpha:.2f})",
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

        # High-Res Download Config
        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "chisq_independence_result",
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

    def _render_results(self, calc: ChiSquareIndepCalculator):
        st.subheader("Statistical Decision")
        col1, col2, col3 = st.columns(3)

        col1.metric("Chi-Square Stat", f"{calc.chi2_stat:.3f}")
        col2.metric("Critical Value", f"> {calc.chi2_critical:.3f}")

        if calc.is_rejected():
            decision_text = "REJECT H₀"
            detail_text = "Significant association detected."
            delta_color = "inverse"
        else:
            decision_text = "FAIL TO REJECT H₀"
            detail_text = "Variables appear independent."
            delta_color = "normal"

        col3.metric(
            "Conclusion",
            decision_text,
            f"p = {calc.p_value:.4f}",
            delta_color=delta_color,
        )

        # Detailed Expected vs Observed Table
        st.markdown("### Expected Frequencies ($E_{ij}$)")
        st.caption(
            "Values in parentheses are the expected counts under Independence ($H_0$)."
        )

        # Format a DataFrame to show Observed (Expected)
        display_data = calc.p.observed_df.copy().astype(str)
        for col in calc.p.observed_df.columns:
            col_idx = calc.p.observed_df.columns.get_loc(col)
            for row_idx, row_name in enumerate(calc.p.observed_df.index):
                obs = calc.observed[row_idx, col_idx]
                exp = calc.expected[row_idx, col_idx]
                display_data.iloc[row_idx, col_idx] = f"{obs} ({exp:.1f})"

        st.dataframe(display_data, use_container_width=True)

        if calc.is_rejected():
            st.error(
                f"**Action Required:** {detail_text} Choice of Language influences OS choice."
            )
        else:
            st.success(
                f"**Status:** {detail_text} Language choice does not predict OS choice."
            )


# --- 3. Entry Point ---
if __name__ == "__main__":
    app = StreamlitHypothesisApp()
    app.run()
