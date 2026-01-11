import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from scipy.special import expit

# --- 1. DATA GENERATION (GOD MODE) ---


class DataEngine:
    def __init__(self, n, b0, b1, seed=42):
        self.rng = np.random.default_rng(seed)
        # Generate X roughly between -3 and 3
        self.X = np.sort(self.rng.uniform(-3, 3, n))
        self.b0 = b0
        self.b1 = b1

    def generate(self, family_type):
        """
        Generates data and returns:
        1. X, y arrays
        2. LaTeX string for the TRUE generating formula
        """
        eta = self.b0 + self.b1 * self.X  # Linear Predictor

        if family_type == "Logistic (Binary)":
            # Link: Logit. Inverse: Sigmoid.
            mu = expit(eta)
            y = self.rng.binomial(1, mu)

            # True Formula string
            formula = rf"\ln\left(\frac{{P}}{{1-P}}\right) = {self.b0} + {self.b1}X"
            ylabel = "Probability P(Y=1)"

        else:  # Poisson
            # Link: Log. Inverse: Exp.
            # Clip eta to avoid overflow in visualization (e^5 is already ~150)
            mu = np.exp(np.clip(eta, -5, 5))
            y = self.rng.poisson(mu)

            formula = rf"\ln(\mu) = {self.b0} + {self.b1}X"
            ylabel = "Count E[Y]"

        return self.X, y, formula, ylabel


# --- 2. MODELING (ANALYST MODE) ---


class ModelEngine:
    @staticmethod
    def fit_and_predict(X, y, family_type):
        X_const = sm.add_constant(X)
        X_grid = np.linspace(X.min(), X.max(), 200)
        X_grid_const = sm.add_constant(X_grid)

        # A. Ordinary Least Squares (The Wrong Model)
        ols = sm.OLS(y, X_const).fit()
        ols_pred = ols.get_prediction(X_grid_const).summary_frame(alpha=0.05)

        # OLS Formula String
        ols_formula = rf"\hat{{y}} = {ols.params[0]:.2f} + {ols.params[1]:.2f}X"

        # B. Generalized Linear Model (The Correct Model)
        if family_type == "Logistic (Binary)":
            glm = sm.GLM(y, X_const, family=sm.families.Binomial()).fit()
            link_str = r"\ln\left(\frac{\hat{p}}{1-\hat{p}}\right)"
        else:
            glm = sm.GLM(y, X_const, family=sm.families.Poisson()).fit()
            link_str = r"\ln(\hat{\mu})"

        glm_pred = glm.get_prediction(X_grid_const).summary_frame(alpha=0.05)
        glm_formula = rf"{link_str} = {glm.params[0]:.2f} + {glm.params[1]:.2f}X"

        return {
            "x_grid": X_grid,
            "ols": {"model": ols, "pred": ols_pred, "eqn": ols_formula},
            "glm": {"model": glm, "pred": glm_pred, "eqn": glm_formula},
        }


# --- 3. VISUALIZATION ---


class Plotter:
    @staticmethod
    def plot_comparison(X, y, results, family_type, ylabel):
        fig = go.Figure()

        # --- DANGER ZONES (Visualizing why OLS fails) ---
        if family_type == "Logistic (Binary)":
            # Prob > 1
            fig.add_shape(
                type="rect",
                x0=X.min(),
                x1=X.max(),
                y0=1.0,
                y1=1.5,
                fillcolor="rgba(255,0,0,0.1)",
                line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=X.min() + 0.5,
                y=1.05,
                text="Impossible (>1)",
                showarrow=False,
                font=dict(color="red"),
            )
            # Prob < 0
            fig.add_shape(
                type="rect",
                x0=X.min(),
                x1=X.max(),
                y0=-0.5,
                y1=0.0,
                fillcolor="rgba(255,0,0,0.1)",
                line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=X.max() - 0.5,
                y=-0.05,
                text="Impossible (<0)",
                showarrow=False,
                font=dict(color="red"),
            )

            # Jitter data for visibility
            y_viz = y + np.random.uniform(-0.02, 0.02, len(y))
            yrange = [-0.2, 1.2]
        else:
            # Count < 0
            fig.add_shape(
                type="rect",
                x0=X.min(),
                x1=X.max(),
                y0=-10,
                y1=0,
                fillcolor="rgba(255,0,0,0.1)",
                line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=X.min() + 0.5,
                y=-0.5,
                text="Impossible (<0)",
                showarrow=False,
                font=dict(color="red"),
            )

            y_viz = y
            # Auto-scale Y but keep negative visible if OLS goes there
            ymax = max(y.max(), results["ols"]["pred"]["mean"].max()) * 1.1
            ymin = min(-2, results["ols"]["pred"]["mean"].min())
            yrange = [ymin, ymax]

        # 1. Observed Data
        fig.add_trace(
            go.Scatter(
                x=X,
                y=y_viz,
                mode="markers",
                marker=dict(color="black", opacity=0.5, size=6),
                name="Observed Data",
            )
        )

        # 2. OLS Fit (Red)
        fig.add_trace(
            go.Scatter(
                x=results["x_grid"],
                y=results["ols"]["pred"]["mean"],
                mode="lines",
                line=dict(color="red", width=3, dash="dash"),
                name="Linear Regression (OLS)",
            )
        )

        # 3. GLM Fit (Blue)
        fig.add_trace(
            go.Scatter(
                x=results["x_grid"],
                y=results["glm"]["pred"]["mean"],
                mode="lines",
                line=dict(color="blue", width=4),
                name="GLM Fit",
            )
        )

        # GLM Confidence Interval
        x_grid = results["x_grid"]
        y_upper = results["glm"]["pred"]["mean_ci_upper"]
        y_lower = results["glm"]["pred"]["mean_ci_lower"]

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_grid, x_grid[::-1]]),
                y=np.concatenate([y_upper, y_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(0,0,255,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name="GLM 95% CI",
            )
        )

        fig.update_layout(
            title=f"Visual Showdown: {family_type}",
            xaxis_title="Input Feature X",
            yaxis_title=ylabel,
            yaxis=dict(range=yrange),
            template="plotly_white",
            height=500,
        )
        return fig


# --- 4. MAIN APP ---


def main():
    st.set_page_config(page_title="GLM Lab", layout="wide")
    st.title("GLM vs. Linear Regression: The Formula Lab")

    st.markdown("""
    Use the sidebar to control the **True Generating Process**. 
    Observe how the **Estimated Equations** change and how OLS fails to respect the data boundaries.
    """)

    # --- SIDEBAR: PARAMETERS ---
    st.sidebar.header("1. Data Configuration")
    family = st.sidebar.radio(
        "Distribution Family", ["Logistic (Binary)", "Poisson (Counts)"]
    )
    n = st.sidebar.slider("Sample Size (N)", 50, 500, 200)

    st.sidebar.divider()
    st.sidebar.header("2. True Coefficients")
    b0 = st.sidebar.slider("Intercept (Beta 0)", -3.0, 3.0, 0.0, step=0.1)
    b1 = st.sidebar.slider("Slope (Beta 1)", -3.0, 3.0, 1.5, step=0.1)

    if st.sidebar.button("Regenerate Random Noise"):
        seed = np.random.randint(0, 1000)
    else:
        seed = 42

    # --- CALCULATIONS ---
    # 1. Generate Truth
    data_eng = DataEngine(n, b0, b1, seed)
    X, y, true_eqn, ylabel = data_eng.generate(family)

    # 2. Estimate Models
    res = ModelEngine.fit_and_predict(X, y, family)

    # --- DISPLAY: EQUATION BOARD ---
    st.subheader("Formula Comparison")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("### 1. The Truth")
        st.caption("This is how the data was actually created.")
        st.latex(true_eqn)

    with c2:
        st.success("### 2. GLM Estimate (Correct)")
        st.caption("The model recovers the truth via the correct Link Function.")
        st.latex(res["glm"]["eqn"])

    with c3:
        st.error("### 3. OLS Estimate (Incorrect)")
        st.caption("Standard regression forces a straight line equation.")
        st.latex(res["ols"]["eqn"])

    st.divider()

    # --- DISPLAY: VISUAL SHOWDOWN ---
    col_viz, col_stats = st.columns([2, 1])

    with col_viz:
        st.plotly_chart(
            Plotter.plot_comparison(X, y, res, family, ylabel), use_container_width=True
        )

    with col_stats:
        st.subheader("Why OLS Fails")
        if family == "Logistic (Binary)":
            st.warning("""
            **1. Invalid Predictions:**
            OLS predicts probability > 1 or < 0 (see Red Zones).
            
            **2. Wrong Shape:**
            Probability follows an S-Curve. OLS forces a straight line.
            """)
        else:
            st.warning("""
            **1. Negative Counts:**
            OLS predicts negative events (see Red Zones).
            
            **2. Variance Mismatch:**
            OLS assumes error variance is constant. In Poisson, variance grows with the mean.
            """)

        st.markdown("---")
        st.metric(
            "GLM Pseudo-R²",
            f"{res['glm']['model'].pseudo_rsquared(kind='mcfadden'):.3f}",
        )
        st.metric(
            "OLS R²",
            f"{res['ols']['model'].rsquared:.3f}",
            delta="Misleading",
            delta_color="inverse",
        )


if __name__ == "__main__":
    main()
