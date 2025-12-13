import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go

st.set_page_config(page_title="Bayesian Coin Flipper", page_icon="🪙", layout="wide")


class BayesianCoinModel:
    """
    Handles the Bayesian update logic for a coin flip experiment
    using the Beta-Binomial conjugate pair.
    """

    def __init__(self, prior_alpha=1, prior_beta=1):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def update(self, heads, tails):
        post_alpha = self.prior_alpha + heads
        post_beta = self.prior_beta + tails
        return post_alpha, post_beta

    @staticmethod
    def get_pdf(alpha_param, beta_param, x_grid):
        return beta.pdf(x_grid, alpha_param, beta_param)

    @staticmethod
    def get_stats(alpha_param, beta_param):
        mean = alpha_param / (alpha_param + beta_param)
        lower, upper = beta.interval(0.95, alpha_param, beta_param)
        return mean, lower, upper


def plot_distributions(x, prior_y, post_y, true_bias, show_truth=False):
    """
    Creates an interactive Plotly chart.
    The 'True Bias' line is now conditional on 'show_truth'.
    """
    fig = go.Figure()

    # 1. Prior Distribution
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prior_y,
            mode="lines",
            name="Prior (Previous Belief)",
            line=dict(color="gray", width=2, dash="dash"),
        )
    )

    # 2. Posterior Distribution
    fig.add_trace(
        go.Scatter(
            x=x,
            y=post_y,
            mode="lines",
            name="Posterior (Updated Belief)",
            line=dict(color="#636EFA", width=3),
            fill="tozeroy",
            fillcolor="rgba(99, 110, 250, 0.2)",
        )
    )

    # 3. True Bias (Conditional)
    if show_truth:
        fig.add_vline(
            x=true_bias,
            line_width=3,
            line_dash="dot",
            line_color="red",
            annotation_text=f"True Bias = {true_bias:.2f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="<b>Belief Distribution: What is the bias?</b>",
        xaxis_title="Bias $p$ (Probability of Heads)",
        yaxis_title="Density (Confidence)",
        xaxis_range=([0, 1]),
        template="plotly_white",
        hovermode="x",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


# --- MAIN APP LOGIC ---
def main():
    # --- Session State Initialization ---
    if "heads" not in st.session_state:
        st.session_state["heads"] = 0
    if "tails" not in st.session_state:
        st.session_state["tails"] = 0

    # Initialize True Bias randomly if not present
    if "true_bias" not in st.session_state:
        # Pick a random bias between 0.1 and 0.9
        st.session_state["true_bias"] = np.random.uniform(0.1, 0.9)

    # Initialize Visibility State
    if "reveal_truth" not in st.session_state:
        st.session_state["reveal_truth"] = False

    # --- Sidebar Controls ---
    st.sidebar.header("1. Setup Experiment")

    # Logic to pick a new secret coin
    if st.sidebar.button("✨ Pick New Secret Coin", type="primary"):
        st.session_state["true_bias"] = np.random.uniform(0.1, 0.9)
        st.session_state["heads"] = 0
        st.session_state["tails"] = 0
        st.session_state["reveal_truth"] = False  # Re-hide the truth
        st.rerun()

    # Logic to reveal the coin
    st.sidebar.divider()
    if st.session_state["reveal_truth"]:
        st.sidebar.success(
            f"**True Bias Revealed:** {st.session_state['true_bias']:.3f}"
        )
        if st.sidebar.button("Hide Answer"):
            st.session_state["reveal_truth"] = False
            st.rerun()
    else:
        st.sidebar.warning("**True Bias is Hidden**")
        if st.sidebar.button("👀 Reveal Answer"):
            st.session_state["reveal_truth"] = True
            st.rerun()

    # --- Prior Selection ---
    st.sidebar.divider()
    st.sidebar.header("2. Choose your Prior")
    prior_type = st.sidebar.selectbox(
        "Initial Belief Strategy",
        ["Uninformed (Uniform)", "Believes Fair", "Skeptical of Fair"],
    )

    if prior_type == "Uninformed (Uniform)":
        alpha_init, beta_init = 1, 1
    elif prior_type == "Believes Fair":
        alpha_init, beta_init = 10, 10
    else:
        alpha_init, beta_init = 0.5, 0.5

    st.sidebar.info(f"Prior: Beta({alpha_init}, {beta_init})")

    # --- Coin Flipping Actions ---
    st.sidebar.divider()
    st.sidebar.header("3. Flip Coin")

    col1, col2 = st.sidebar.columns(2)
    # Note: We must use the stored session_state bias, not a slider value!
    current_true_bias = st.session_state["true_bias"]

    if col1.button("Flip (1x)"):
        outcome = np.random.binomial(n=1, p=current_true_bias)
        if outcome == 1:
            st.session_state["heads"] += 1
        else:
            st.session_state["tails"] += 1

    if col2.button("Flip (10x)"):
        outcomes = np.random.binomial(n=10, p=current_true_bias)
        st.session_state["heads"] += outcomes
        st.session_state["tails"] += 10 - outcomes

    # --- Main Page Content ---
    st.title("Bayesian Inference: Guess the Bias")
    st.markdown("""
    A magician has handed you a coin. It might be fair, or it might be tricked.
    **You don't know the bias.**

    Flip the coin, observe the Posterior update, and try to guess where the peak is before revealing the answer!
    """)

    # --- Calculations ---
    model = BayesianCoinModel(alpha_init, beta_init)
    curr_h = st.session_state["heads"]
    curr_t = st.session_state["tails"]
    total_flips = curr_h + curr_t

    post_alpha, post_beta = model.update(curr_h, curr_t)
    x = np.linspace(0, 1, 500)

    prior_pdf = model.get_pdf(alpha_init, beta_init, x)
    post_pdf = model.get_pdf(post_alpha, post_beta, x)
    mean_est, cred_lower, cred_upper = model.get_stats(post_alpha, post_beta)

    # --- Metrics ---
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Flips", f"{total_flips}")
    m2.metric("Heads", f"{curr_h}")
    m3.metric("Tails", f"{curr_t}")
    m4.metric("Est. Bias (Mean)", f"{mean_est:.3f}")

    # --- Plot ---
    # We pass the reveal_truth flag here
    fig = plot_distributions(
        x,
        prior_pdf,
        post_pdf,
        current_true_bias,
        show_truth=st.session_state["reveal_truth"],
    )
    st.plotly_chart(fig, width="stretch")

    # --- Explanation ---
    st.subheader("Your Bayesian Update")
    st.markdown(f"""
    * **Prior:** $Beta({alpha_init}, {beta_init})$
    * **Data:** {curr_h}H, {curr_t}T
    * **Posterior:** $Beta({post_alpha}, {post_beta})$

    Based on the data, there is a 95% chance the bias is between **{cred_lower:.2f}** and **{cred_upper:.2f}**.
    """)

    if st.session_state["reveal_truth"]:
        error = abs(current_true_bias - mean_est)
        st.info(
            f"The True Bias was **{current_true_bias:.3f}**. Your estimate error was **{error:.3f}**."
        )


if __name__ == "__main__":
    main()
