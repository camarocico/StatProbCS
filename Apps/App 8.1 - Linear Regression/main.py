import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass
from typing import Tuple, Literal

# --- 1. MODEL LAYER ---


@dataclass
class OLSMetrics:
    mse: float
    rmse: float
    r2: float
    correlation: float
    slope: float
    intercept: float


class DataGenerator:
    """
    Generates synthetic data and returns the 'True' underlying equation string.
    """

    @staticmethod
    def get_data(
        n: int, noise_level: float, pattern: str, seed: int
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 10, n).reshape(-1, 1)
        noise = rng.normal(0, noise_level, n)

        # Ground Truth Parameters
        beta_1 = 2.0
        beta_0 = 5.0

        if pattern == "Linear (Ideal)":
            y = (beta_1 * X.flatten()) + beta_0 + noise
            eq_str = f"y = {beta_1}x + {beta_0} + \epsilon"

        elif pattern == "Quadratic (Violates Linearity)":
            # y = 0.5x^2 + 5 + noise
            y = (0.5 * (X.flatten() ** 2)) + beta_0 + noise
            eq_str = f"y = 0.5x^2 + {beta_0} + \epsilon"

        elif pattern == "Heteroscedastic (Violates Homoscedasticity)":
            # Noise scales with X
            expanding_noise = noise * (X.flatten() * 0.8)
            y = (beta_1 * X.flatten()) + beta_0 + expanding_noise
            eq_str = f"y = {beta_1}x + {beta_0} + \epsilon(x)"

        return X, y, eq_str


class RegressionModel:
    """Handles OLS fitting and metric extraction."""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.y_pred = self.model.predict(X)
        self.residuals = y - self.y_pred

    def get_metrics(self) -> OLSMetrics:
        mse = mean_squared_error(self.y, self.y_pred)
        correlation = np.corrcoef(self.X.flatten(), self.y)[0, 1]

        return OLSMetrics(
            mse=mse,
            rmse=np.sqrt(mse),
            r2=r2_score(self.y, self.y_pred),
            correlation=correlation,
            slope=self.model.coef_[0],
            intercept=self.model.intercept_,
        )


# --- 2. VIEW LAYER ---


class Plotter:
    @staticmethod
    def create_dashboard(X, y, y_pred, residuals, pattern_type):
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[0.7, 0.3],
            specs=[[{"rowspan": 2}, {}], [None, {}]],
            subplot_titles=(
                "Regression Analysis",
                "Residuals vs Fitted",
                "Residual Histogram",
            ),
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
        )

        # 1. Main Plot
        fig.add_trace(
            go.Scatter(
                x=X.flatten(),
                y=y,
                mode="markers",
                name="Observed Data",
                marker=dict(color="black", opacity=0.5, size=7),
            ),
            row=1,
            col=1,
        )

        sort_idx = np.argsort(X.flatten())
        fig.add_trace(
            go.Scatter(
                x=X.flatten()[sort_idx],
                y=y_pred[sort_idx],
                mode="lines",
                name="Estimated Line",
                line=dict(color="blue", width=3),
            ),
            row=1,
            col=1,
        )

        # Residual lines (Red/Green)
        shapes = []
        for xi, yi, yhat in zip(X.flatten(), y, y_pred):
            color = "rgba(255, 0, 0, 0.3)" if yi < yhat else "rgba(0, 128, 0, 0.3)"
            shapes.append(
                dict(
                    type="line",
                    x0=xi,
                    y0=yi,
                    x1=xi,
                    y1=yhat,
                    line=dict(color=color, width=1),
                    layer="below",
                )
            )
        fig.update_layout(shapes=shapes)

        # 2. Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker=dict(color="purple", opacity=0.6),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

        # 3. Histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=15,
                marker_color="purple",
                opacity=0.7,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            template="plotly_white",
            height=600,
            title_text=f"OLS Diagnostics: {pattern_type}",
        )
        return fig


# --- 3. CONTROLLER LAYER ---


def main():
    st.set_page_config(page_title="Regression Lab", layout="wide")

    st.title("Regression Analysis Lab")

    # Sidebar
    st.sidebar.header("Data Configuration")
    data_pattern = st.sidebar.selectbox(
        "Data Pattern",
        [
            "Linear (Ideal)",
            "Quadratic (Violates Linearity)",
            "Heteroscedastic (Violates Homoscedasticity)",
        ],
    )
    n_samples = st.sidebar.slider("Sample Size ($n$)", 20, 500, 100)
    noise = st.sidebar.slider("Noise Level ($\sigma$)", 0.1, 10.0, 3.0)

    if st.sidebar.button("Regenerate Random Data"):
        seed = np.random.randint(0, 1000)
    else:
        seed = 42

    # Logic
    X, y, true_eqn = DataGenerator.get_data(n_samples, noise, data_pattern, seed)
    model = RegressionModel(X, y)
    metrics = model.get_metrics()

    # --- TOP SECTION: EQUATION COMPARISON ---
    st.markdown("### Model vs. Truth")
    eq_col1, eq_col2, eq_col3 = st.columns([1, 1, 2])

    with eq_col1:
        st.info("**True Generation Process**")
        st.latex(true_eqn)

    with eq_col2:
        st.success("**Estimated OLS Model**")
        # Construct the fitted equation string dynamically
        sign = "+" if metrics.intercept >= 0 else "-"
        st.latex(
            f"\hat{{y}} = {metrics.slope:.2f}x {sign} {abs(metrics.intercept):.2f}"
        )

    with eq_col3:
        # Highlight the difference in slope
        # Only meaningful if the model is actually linear
        if "Linear" in data_pattern:
            error = abs(metrics.slope - 2.0)
            st.metric(
                label="Estimation Error (Slope)",
                value=f"{error:.3f}",
                delta="Low Error" if error < 0.1 else "High Error",
                delta_color="inverse",
                help="Difference between True Slope (2.0) and Estimated Slope",
            )
        else:
            st.warning("Cannot compare slopes directly: True model is non-linear.")

    st.divider()

    # --- DASHBOARD & METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Correlation ($r$)", f"{metrics.correlation:.3f}")
    with m2:
        st.metric("$R^2$", f"{metrics.r2:.3f}")
    with m3:
        st.metric("RMSE", f"{metrics.rmse:.2f}")
    with m4:
        st.metric("Sample Size", f"{n_samples}")

    st.plotly_chart(
        Plotter.create_dashboard(X, y, model.y_pred, model.residuals, data_pattern),
        use_container_width=True,
    )

    # --- EDUCATIONAL FOOTER ---
    st.divider()
    if data_pattern == "Linear (Ideal)":
        st.caption(
            "✅ **Analysis:** The estimated equation is structure-correct. As $n \\to \infty$, $\hat{\\beta}$ converges to $\\beta$."
        )
    elif "Quadratic" in data_pattern:
        st.caption(
            "❌ **Analysis:** The estimator assumes a line $y=mx+c$, but the truth is quadratic. The model is **misspecified**."
        )


if __name__ == "__main__":
    main()
