import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 1. DATA GENERATION ENGINE ---


class TrueModel:
    """
    Manages the 'God Mode' parameters and generates data based on exact specifications.
    """

    def __init__(self, n, noise, b0, b1, b2, b_int, add_collinear, add_omitted):
        self.n = n
        self.noise = noise
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b_int = b_int
        self.add_collinear = add_collinear
        self.add_omitted = add_omitted

    def generate(self):
        rng = np.random.default_rng(42)

        # 1. Independent Variables
        x1 = rng.uniform(0, 10, self.n)
        x2 = rng.uniform(0, 10, self.n)
        epsilon = rng.normal(0, self.noise, self.n)

        # 2. Logic for Omitted Variable Bias (Endogeneity)
        if self.add_omitted:
            # Z is the hidden confounder. It correlates with X1.
            z = x1 + rng.normal(0, 2.0, self.n)
            # Z affects Y, but will NOT be in the output dataframe
            z_effect = 3.0 * z
        else:
            z_effect = 0

        # 3. The True Equation
        y = (
            self.b0
            + (self.b1 * x1)
            + (self.b2 * x2)
            + (self.b_int * x1 * x2)
            + z_effect
            + epsilon
        )

        # 4. Construct Observed Data
        df = pd.DataFrame({"x1": x1, "x2": x2})

        if self.add_collinear:
            # X3 is a near-perfect copy of X1
            df["x3"] = x1 + rng.normal(0, 0.5, self.n)

        return df, y

    def get_latex_equation(self):
        """Returns the LaTeX string of the TRUE process."""
        eq = f"y = {self.b0} + {self.b1}x_1 + {self.b2}x_2"

        if self.b_int != 0:
            eq += f" + {self.b_int}(x_1 \\cdot x_2)"

        if self.add_omitted:
            eq += f" + 3.0(Z_{{hidden}})"

        eq += f" + \epsilon"
        return eq


# --- 2. VISUALIZATION HELPERS ---


def plot_3d_comparison(df, y, model, title):
    """
    Visualizes the raw data (cloud) vs the fitted model (plane/surface).
    """
    # Create meshgrid for the surface
    x1_range = np.linspace(df["x1"].min(), df["x1"].max(), 20)
    x2_range = np.linspace(df["x2"].min(), df["x2"].max(), 20)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

    # Predict Z values for the surface
    # We construct a synthetic dataframe for the prediction that matches model structure
    pred_data = pd.DataFrame(
        {"const": 1.0, "x1": x1_mesh.ravel(), "x2": x2_mesh.ravel()}
    )

    # Handle Interaction Term in visualization
    if "x1:x2" in model.params.index:
        pred_data["x1:x2"] = pred_data["x1"] * pred_data["x2"]

    # Handle Collinear Term in visualization (Assumption: x3 ≈ x1)
    if "x3" in model.params.index:
        pred_data["x3"] = pred_data["x1"]

    z_pred = model.predict(pred_data).values.reshape(x1_mesh.shape)

    fig = go.Figure()

    # Raw Data
    fig.add_trace(
        go.Scatter3d(
            x=df["x1"],
            y=df["x2"],
            z=y,
            mode="markers",
            marker=dict(size=4, color="black", opacity=0.4),
            name="Observed Data",
        )
    )

    # Regression Surface
    fig.add_trace(
        go.Surface(
            x=x1_range,
            y=x2_range,
            z=z_pred,
            colorscale="Viridis",
            opacity=0.8,
            name="Model Fit",
            showscale=False,
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X1", yaxis_title="X2", zaxis_title="Y"),
        margin=dict(l=0, r=0, b=0, t=30),
        height=500,
    )
    return fig


def plot_coefficients(model, true_b1, true_b2):
    """
    Visualizes Estimated Coefficients vs True Coefficients with Confidence Intervals.
    """
    params = model.params.drop("const")
    conf = model.conf_int().drop("const")

    # Calculate error bar size
    errors = params - conf[0]

    fig = go.Figure()

    # Estimates
    fig.add_trace(
        go.Bar(
            x=params.index,
            y=params.values,
            error_y=dict(type="data", array=errors.values, visible=True, color="red"),
            name="Estimate",
            marker_color="blue",
            opacity=0.6,
        )
    )

    # True Values Markers (Only for x1 and x2 for clarity)
    if "x1" in params.index:
        fig.add_trace(
            go.Scatter(
                x=["x1"],
                y=[true_b1],
                mode="markers",
                marker=dict(color="green", size=15, symbol="star"),
                name="True Beta 1",
            )
        )
    if "x2" in params.index:
        fig.add_trace(
            go.Scatter(
                x=["x2"],
                y=[true_b2],
                mode="markers",
                marker=dict(color="green", size=15, symbol="star"),
                name="True Beta 2",
            )
        )

    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title="Estimates vs. Truth (Green Star)", yaxis_title="Coefficient Value"
    )
    return fig


# --- 3. MAIN APPLICATION ---


def main():
    st.set_page_config(layout="wide", page_title="Regression Sandbox")
    st.title("🎛️ Linear Regression Sandbox")
    st.markdown(
        "Define the **True Generating Process** (God Mode) and see if your **Statistical Model** can recover it."
    )

    # --- SIDEBAR: GOD MODE (Data Generation) ---
    st.sidebar.header("1. True Process")
    st.sidebar.caption("This controls how the data is actually created.")

    # True Coefficients
    n_samples = st.sidebar.slider("Sample Size ($N$)", 50, 500, 200)
    noise = st.sidebar.slider("Noise ($\sigma$)", 0.0, 10.0, 3.0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**True Coefficients ($\\beta$)**")
    true_b0 = st.sidebar.slider(
        "Intercept ($\\beta_0$)", -5.0, 5.0, value=5.0, step=0.1
    )
    true_b1 = st.sidebar.slider("Slope $X_1$ ($\\beta_1$)", -5.0, 5.0, 2.0)
    true_b2 = st.sidebar.slider("Slope $X_2$ ($\\beta_2$)", -5.0, 5.0, 1.0)
    true_int = st.sidebar.slider(
        "Interaction ($\\beta_{1,2}$)",
        -2.0,
        2.0,
        0.0,
        help="Synergy effect between X1 and X2",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Challenges**")
    add_collinear = st.sidebar.checkbox(
        "Add Collinear Variable ($X_3 \\approx X_1$)", value=False
    )
    add_omitted = st.sidebar.checkbox(
        "Add Omitted Variable ($Z$)",
        value=False,
        help="A hidden variable Z correlated with X1 affects Y.",
    )

    # --- SIDEBAR: ANALYST MODE (Model Spec) ---
    st.sidebar.header("2. Your Model (Analyst Mode)")
    st.sidebar.caption("Which variables do you want to include in the regression?")

    use_x1 = st.sidebar.checkbox(
        "Include $X_1$", value=True, disabled=True
    )  # Always included base
    use_x2 = st.sidebar.checkbox("Include $X_2$", value=True)
    use_int = st.sidebar.checkbox("Include Interaction ($X_1 \cdot X_2$)", value=False)

    use_x3 = False
    if add_collinear:
        use_x3 = st.sidebar.checkbox("Include $X_3$ (Collinear)", value=True)

    # --- GENERATE DATA ---
    generator = TrueModel(
        n_samples,
        noise,
        true_b0,
        true_b1,
        true_b2,
        true_int,
        add_collinear,
        add_omitted,
    )
    df, y = generator.generate()

    # --- FIT MODEL ---
    # Construct Design Matrix based on Analyst choices
    X_design = df.copy()
    if not use_x3 and "x3" in X_design.columns:
        X_design = X_design.drop(columns=["x3"])
    if not use_x2:
        X_design = X_design.drop(columns=["x2"])

    # Interaction term creation (Feature Engineering)
    if use_int:
        X_design["x1:x2"] = X_design["x1"] * X_design["x2"]

    X_design = sm.add_constant(X_design)
    model = sm.OLS(y, X_design).fit()

    # --- DASHBOARD ---

    # 1. Equation Comparison
    col_truth, col_est = st.columns(2)
    with col_truth:
        st.info("### True Process (Hidden from Analyst)")
        st.latex(generator.get_latex_equation())

    with col_est:
        color = "green" if model.rsquared > 0.8 else "red"
        st.markdown(f"### Estimated Model (:{color}[$R^2 = {model.rsquared:.2f}$])")

        # Build Estimated LaTeX String dynamically
        est_eq = f"\hat{{y}} = {model.params['const']:.2f}"
        for col in model.params.index:
            if col != "const":
                est_eq += f" + {model.params[col]:.2f}({col})"
        st.latex(est_eq)

    # 2. Visuals
    tab1, tab2, tab3 = st.tabs(
        ["3D Fit Visualization", "Coefficient Integrity", "Raw Stats"]
    )

    with tab1:
        st.plotly_chart(
            plot_3d_comparison(df, y, model, "Model Fit Surface"),
            use_container_width=True,
        )

    with tab2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(
                plot_coefficients(model, true_b1, true_b2), use_container_width=True
            )
        with c2:
            if add_collinear and use_x3:
                st.error("⚠️ **Multicollinearity Warning**")
                st.write(
                    "Notice how the Error Bars (Red Lines) on $X_1$ and $X_3$ are huge? The model is unsure how to split the credit."
                )
            elif add_omitted:
                st.error("⚠️ **Omitted Variable Bias**")
                st.write(
                    f"True $\\beta_1$ is {true_b1}, but estimated is {model.params['x1']:.2f}. The model is over-crediting $X_1$ because it can't see $Z$."
                )
            elif true_int != 0 and not use_int:
                st.warning("⚠️ **Underfitting**")
                st.write(
                    "The True Process has an interaction, but your model is linear. The estimates are biased."
                )
            else:
                st.success("✅ **Robust Estimation**")
                st.write(
                    "The estimates are close to the Green Stars (Truth) and error bars are narrow."
                )

    with tab3:
        st.write(model.summary())


if __name__ == "__main__":
    main()
