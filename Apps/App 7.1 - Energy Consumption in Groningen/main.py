import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass

# --- 1. Model Layer: Statistical Logic ---


@dataclass
class TestResult:
    """Data transfer object for test results."""

    se: float
    z_score: float
    p_value: float
    critical_value_low: float
    critical_value_high: float
    reject_null: bool


class ZTestModel:
    """
    Handles the mathematical logic for a two-tailed Z-test.
    """

    def __init__(self, mu_0: float, sigma: float, n: int, x_bar: float, alpha: float):
        self.mu_0 = mu_0  # Hypothesized mean
        self.sigma = sigma  # Population std dev
        self.n = n  # Sample size
        self.x_bar = x_bar  # Sample mean
        self.alpha = alpha  # Significance level

    def calculate(self) -> TestResult:
        """Performs the statistical calculation."""
        # 1. Standard Error
        se = self.sigma / np.sqrt(self.n)

        # 2. Z-Score
        z_score = (self.x_bar - self.mu_0) / se

        # 3. Critical Values (X-scale)
        # We find the Z critical values first (+/- 1.96 for alpha=0.05) then convert to X scale
        z_crit = norm.ppf(1 - self.alpha / 2)
        x_crit_low = self.mu_0 - (z_crit * se)
        x_crit_high = self.mu_0 + (z_crit * se)

        # 4. P-Value (Two-tailed)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # 5. Decision
        reject = p_value < self.alpha

        return TestResult(se, z_score, p_value, x_crit_low, x_crit_high, reject)


# --- 2. View Layer: Visualization ---


class PlotlyVisualizer:
    """
    Handles the creation of the interactive Plotly chart.
    """

    @staticmethod
    def create_sampling_distribution_plot(model: ZTestModel, result: TestResult):
        # Generate X values for the plot (spanning 4 standard errors)
        x = np.linspace(model.mu_0 - 4 * result.se, model.mu_0 + 4 * result.se, 1000)
        y = norm.pdf(x, model.mu_0, result.se)

        fig = go.Figure()

        # 1. The H0 Distribution Curve
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="H0 Distribution",
                line=dict(color="gray", width=2),
                hovertemplate="<b>H₀ Distribution</b><br>Sample Mean (x̄): %{x:.2f} kWh<br>Density: %{y:.4f}<extra></extra>",
            )
        )

        # 2. Critical Regions (Rejection Zones) - Left Tail
        x_left = np.linspace(x[0], result.critical_value_low, 100)
        y_left = norm.pdf(x_left, model.mu_0, result.se)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(([x_left[0]], x_left, [x_left[-1]])),
                y=np.concatenate(([0], y_left, [0])),
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.2)",
                line=dict(width=0),
                name="Rejection Region",
                hovertemplate="<b>Rejection Region (Left)</b><br>Sample Mean (x̄): %{x:.2f} kWh<extra></extra>",
            )
        )

        # 2. Critical Regions (Rejection Zones) - Right Tail
        x_right = np.linspace(result.critical_value_high, x[-1], 100)
        y_right = norm.pdf(x_right, model.mu_0, result.se)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(([x_right[0]], x_right, [x_right[-1]])),
                y=np.concatenate(([0], y_right, [0])),
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.2)",
                line=dict(width=0),
                name="Rejection Region",
                showlegend=False,
                hovertemplate="<b>Rejection Region (Right)</b><br>Sample Mean (x̄): %{x:.2f} kWh<extra></extra>",
            )
        )

        # 3. Critical Value Lines
        max_y = max(y)
        fig.add_vline(
            x=result.critical_value_low, line_dash="dash", line_color="red", opacity=0.5
        )
        fig.add_vline(
            x=result.critical_value_high,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
        )

        # 4. Observed Sample Mean Line
        fig.add_vline(x=model.x_bar, line_color="blue", line_width=3)
        fig.add_annotation(
            x=model.x_bar,
            y=max_y * 0.8,
            text=f"Observed x̄ = {model.x_bar}",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40,
            bgcolor="white",
            bordercolor="blue",
        )

        # Layout styling
        fig.update_layout(
            title="Sampling Distribution of the Mean under H₀",
            xaxis_title="Average Electricity Consumption (kWh)",
            yaxis_title="Probability Density",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        return fig


# --- 3. Controller Layer: Streamlit UI ---


class HypothesisTestingApp:
    """
    Main application class managing the UI and interaction flow.
    """

    def __init__(self):
        st.set_page_config(
            page_title="Hypothesis Testing: Groningen Electricity", layout="wide"
        )
        self.visualizer = PlotlyVisualizer()

    def render_sidebar(self):
        st.sidebar.header("Experimental Parameters")

        st.sidebar.markdown("### Null Hypothesis ($H_0$)")
        mu_0 = st.sidebar.number_input(
            label="Claimed Mean (kWh) [$\\mu_0$]",
            value=500.0,
            step=1.0,
            help="The value claimed by the city council.",
        )

        st.sidebar.markdown("### Population Data")
        sigma = st.sidebar.number_input(
            label="Standard Deviation [$\\sigma$]",
            value=50.0,
            min_value=1.0,
            step=1.0,
            help="The (known) standard deviation of the data",
        )

        st.sidebar.markdown("### Sample Data")
        n = st.sidebar.slider(
            "Sample Size [$n$]", min_value=10, max_value=500, value=100, step=10
        )
        x_bar = st.sidebar.slider(
            "Sample Mean (kWh) [$\\overline{x}$]",
            min_value=450.0,
            max_value=550.0,
            value=515.0,
            step=1.0,
        )

        st.sidebar.markdown("### Test Settings")
        alpha = st.sidebar.selectbox(
            "Significance Level [$\\alpha$]", options=[0.01, 0.05, 0.10], index=1
        )

        return mu_0, sigma, n, x_bar, alpha

    def render_math_explanation(self, model: ZTestModel, result: TestResult):
        st.markdown(r"""### 1. Hypothesis Formulation""")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"H_0: \mu = " + f"{model.mu_0} " + r"\text{ kWh}")
        with col2:
            st.latex(r"H_1: \mu \neq " + f"{model.mu_0} " + r"\text{ kWh}")

        st.markdown(r"""### 2. Calculation""")
        st.write("First, we calculate the Standard Error (SE) and the Z-score.")

        st.latex(
            r"SE = \frac{\sigma}{\sqrt{n}} = \frac{"
            + f"{model.sigma}"
            + r"}{\sqrt{"
            + f"{model.n}"
            + r"}} = "
            + f"{result.se:.2f}"
        )
        st.latex(
            r"Z = \frac{\overline{x} - \mu_0}{SE} = \frac{"
            + f"{model.x_bar} - {model.mu_0}"
            + r"}{"
            + f"{result.se:.2f}"
            + r"} = \mathbf{"
            + f"{result.z_score:.2f}"
            + r"}"
        )

    def render_conclusion(self, result: TestResult, alpha: float):
        st.markdown("### 3. Conclusion")

        result_color = "red" if result.reject_null else "green"
        decision_text = "REJECT" if result.reject_null else "FAIL TO REJECT"

        st.markdown(
            f"""
        The p-value is **{result.p_value:.4f}**.
        
        Since $p$-value {("<" if result.reject_null else ">")} $\\alpha$ ({alpha}):
        
        ### <span style='color:{result_color}'> Decision: {decision_text} the Null Hypothesis</span>
        """,
            unsafe_allow_html=True,
        )

        if result.reject_null:
            st.info(
                "There is sufficient statistical evidence to suggest the actual average consumption differs from the council's claim."
            )
        else:
            st.info(
                "There is NOT sufficient statistical evidence to reject the council's claim. The difference could be due to random sampling chance."
            )

    def run(self):
        st.title("📊 Hypothesis Testing: The Groningen Council Claim")
        st.markdown("""
        **Scenario:** The city council of Groningen claims average household electricity consumption is **500 kWh**. 
        You suspect it is different. You have a sample of data and want to perform a **Two-Tailed Z-Test**.
        """)

        # 1. Get Inputs
        mu_0, sigma, n, x_bar, alpha = self.render_sidebar()

        # 2. Run Model
        model = ZTestModel(mu_0, sigma, n, x_bar, alpha)
        result = model.calculate()

        # 3. Layout: Top Calculation, Bottom Visuals
        col_math, col_viz = st.columns([1, 1.5])

        with col_math:
            self.render_math_explanation(model, result)
            self.render_conclusion(result, alpha)

        with col_viz:
            st.markdown("### Visualizing the Sampling Distribution")
            fig = self.visualizer.create_sampling_distribution_plot(model, result)
            st.plotly_chart(fig, width="stretch")

            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Standard Error", f"{result.se:.2f}")
            m2.metric("Z-Score", f"{result.z_score:.2f}")
            m3.metric(
                "Critical Range (kWh)",
                f"{result.critical_value_low:.1f} - {result.critical_value_high:.1f}",
            )


if __name__ == "__main__":
    app = HypothesisTestingApp()
    app.run()
