import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import beta
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Beta-Binomial Conjugacy Explorer", page_icon="📐", layout="wide"
)


# --- CLASS DEFINITION ---
class BayesianConjugateModel:
    """
    Handles the Bayesian update logic for Beta-Binomial Conjugacy.
    """

    def __init__(self, prior_alpha=1, prior_beta=1):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def update(self, heads, tails):
        return self.prior_alpha + heads, self.prior_beta + tails

    @staticmethod
    def get_pdf(alpha_p, beta_p, x_grid):
        return beta.pdf(x_grid, alpha_p, beta_p)

    @staticmethod
    def get_likelihood(heads, tails, x_grid):
        # Calculate unnormalized likelihood: p^h * (1-p)^t
        lh = (x_grid**heads) * ((1 - x_grid) ** tails)
        # Normalize for visualization only
        if np.sum(lh) > 0:
            lh = lh / (np.sum(lh) * (x_grid[1] - x_grid[0]))
        return lh

    @staticmethod
    def get_summaries(alpha_p, beta_p):
        """
        Returns a dictionary of summary statistics: Mean, MAP, CI.
        """
        # 1. Mean (Expected Value) = alpha / (alpha + beta)
        mean_val = alpha_p / (alpha_p + beta_p)

        # 2. MAP (Mode)
        # Formula: (alpha - 1) / (alpha + beta - 2) for alpha, beta > 1
        if alpha_p > 1 and beta_p > 1:
            map_val = (alpha_p - 1) / (alpha_p + beta_p - 2)
        elif alpha_p == 1 and beta_p == 1:
            map_val = (
                0.5  # Technically undefined (flat), but 0.5 is standard placeholder
            )
        elif alpha_p <= 1 and beta_p > 1:
            map_val = 0.0
        elif alpha_p > 1 and beta_p <= 1:
            map_val = 1.0
        else:
            # U-shaped (bimodal at 0 and 1)
            map_val = 0.0 if alpha_p > beta_p else 1.0  # Simplified return

        # 3. 95% Credible Interval (Equal-tailed)
        lower, upper = beta.interval(0.95, alpha_p, beta_p)

        return {"Mean": mean_val, "MAP": map_val, "CI_Lower": lower, "CI_Upper": upper}


# --- PLOTTING FUNCTIONS ---
def plot_conjugacy_triad(
    x, prior_y, likelihood_y, post_y, true_bias, stats, show_truth=False
):
    fig = go.Figure()

    # 1. Prior
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prior_y,
            name="Prior (Initial)",
            line=dict(color="gray", width=2, dash="dash"),
        )
    )

    # 2. Likelihood
    fig.add_trace(
        go.Scatter(
            x=x,
            y=likelihood_y,
            name="Likelihood (Data)",
            line=dict(color="#00CC96", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 204, 150, 0.1)",
        )
    )

    # 3. Posterior
    fig.add_trace(
        go.Scatter(
            x=x,
            y=post_y,
            name="Posterior (Result)",
            line=dict(color="#636EFA", width=4),
            fill="tozeroy",
            fillcolor="rgba(99, 110, 250, 0.2)",
        )
    )

    # 4. Add MAP line (Dashed Blue)
    fig.add_vline(
        x=stats["MAP"],
        line_width=1,
        line_dash="dash",
        line_color="#636EFA",
        annotation_text="MAP",
        annotation_position="top left",
    )

    # 5. True Bias
    if show_truth:
        fig.add_vline(
            x=true_bias,
            line_width=3,
            line_dash="dot",
            line_color="red",
            annotation_text=f"True p={true_bias:.2f}",
        )

    # Add CI Shading (Optional visual flair - highlight the 95% region)
    # We won't add extra geometry to keep it clean, but we can label the axis

    fig.update_layout(
        title="<b>Conjugate Update: Prior + Likelihood = Posterior</b>",
        xaxis_title="Bias $p$ (Probability of Heads)",
        yaxis_title="Density",
        template="plotly_white",
        hovermode="x",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# --- MAIN APP LOGIC ---
def main():
    # --- State ---
    if "heads" not in st.session_state:
        st.session_state.update({"heads": 0, "tails": 0})
    if "true_bias" not in st.session_state:
        st.session_state["true_bias"] = np.random.uniform(0.1, 0.9)
    if "reveal_truth" not in st.session_state:
        st.session_state["reveal_truth"] = False

    # --- Sidebar ---
    st.sidebar.header("1. Experiment Controls")

    if st.sidebar.button("✨ Pick New Secret Coin"):
        st.session_state["true_bias"] = np.random.uniform(0.1, 0.9)
        st.session_state["heads"] = 0
        st.session_state["tails"] = 0
        st.session_state["reveal_truth"] = False
        st.rerun()

    # Toggle Truth
    if st.session_state["reveal_truth"]:
        if st.sidebar.button("Hide Answer"):
            st.session_state["reveal_truth"] = False
            st.rerun()
    else:
        if st.sidebar.button("👀 Reveal Answer"):
            st.session_state["reveal_truth"] = True
            st.rerun()

    st.sidebar.divider()

    # Prior
    st.sidebar.header("2. Prior Settings")
    prior_mode = st.sidebar.selectbox(
        "Prior Knowledge", ["Uninformed (1,1)", "Fair (10,10)", "Custom"]
    )

    if prior_mode == "Uninformed (1,1)":
        alpha_init, beta_init = 1, 1
    elif prior_mode == "Fair (10,10)":
        alpha_init, beta_init = 10, 10
    else:
        c1, c2 = st.sidebar.columns(2)
        alpha_init = c1.number_input("Alpha", 0.1, 100.0, 1.0, 0.5)
        beta_init = c2.number_input("Beta", 0.1, 100.0, 1.0, 0.5)

    st.sidebar.divider()

    # Flip
    st.sidebar.header("3. Generate Data")
    c1, c2 = st.sidebar.columns(2)
    current_bias = st.session_state["true_bias"]

    if c1.button("Flip 1x"):
        outcome = np.random.binomial(1, current_bias)
        if outcome:
            st.session_state["heads"] += 1
        else:
            st.session_state["tails"] += 1

    if c2.button("Flip 10x"):
        outcomes = np.random.binomial(10, current_bias)
        st.session_state["heads"] += outcomes
        st.session_state["tails"] += 10 - outcomes

    if st.sidebar.button("Reset Flips Only"):
        st.session_state["heads"] = 0
        st.session_state["tails"] = 0

    # --- Model Calculations ---
    model = BayesianConjugateModel(alpha_init, beta_init)
    h_obs, t_obs = st.session_state["heads"], st.session_state["tails"]

    # Update Posterior parameters
    post_alpha, post_beta = model.update(h_obs, t_obs)

    # Get Summaries
    stats = model.get_summaries(post_alpha, post_beta)

    # Grids for plotting
    x = np.linspace(0, 1, 500)
    prior_pdf = model.get_pdf(alpha_init, beta_init, x)
    lik_curve = model.get_likelihood(h_obs, t_obs, x)
    post_pdf = model.get_pdf(post_alpha, post_beta, x)

    # --- UI Layout ---
    st.title("Conjugate Priors: Beta-Binomial Summaries")

    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Heads ($h$)", h_obs)
    m2.metric("Tails ($t$)", t_obs)
    m3.metric("Prior Beta", f"({alpha_init}, {beta_init})")
    m4.metric("Posterior Beta", f"({post_alpha}, {post_beta})")

    # Plot
    fig = plot_conjugacy_triad(
        x,
        prior_pdf,
        lik_curve,
        post_pdf,
        current_bias,
        stats,
        st.session_state["reveal_truth"],
    )
    st.plotly_chart(fig, width="stretch")

    # --- Summary Statistics Section ---
    st.subheader("📊 Posterior Summaries")

    col_text, col_stats = st.columns([1, 1])

    with col_text:
        st.markdown("""
        **1. MAP (Maximum A Posteriori):** The single most likely value (the peak of the posterior curve).
        $$ \\text{MAP} = \\frac{\\alpha_{new} - 1}{\\alpha_{new} + \\beta_{new} - 2} $$
        
        **2. Posterior Mean:**
        The expected value (average).
        $$ \\mathbb{E}[p] = \\frac{\\alpha_{new}}{\\alpha_{new} + \\beta_{new}} $$
        
        **3. 95% Credible Interval (CI):**
        There is a 95% probability that the true bias lies within this range given the data.
        """)

    with col_stats:
        # Create a nice dataframe for display
        df_stats = pd.DataFrame(
            {
                "Metric": ["MAP (Mode)", "Mean", "95% CI Lower", "95% CI Upper"],
                "Value": [
                    f"{stats['MAP']:.4f}",
                    f"{stats['Mean']:.4f}",
                    f"{stats['CI_Lower']:.4f}",
                    f"{stats['CI_Upper']:.4f}",
                ],
            }
        )
        st.table(df_stats)

    # --- Math Derivation ---
    st.divider()
    with st.expander("See Mathematical Derivation"):
        st.markdown("### The Algebra")
        st.latex(rf"""
        \text{{Posterior}} \propto \theta^{{ ({alpha_init} + {h_obs}) - 1 }} (1-\theta)^{{ ({beta_init} + {t_obs}) - 1 }}
        """)
        st.markdown(f"New parameters: $\\alpha' = {post_alpha}, \\beta' = {post_beta}$")


if __name__ == "__main__":
    main()
