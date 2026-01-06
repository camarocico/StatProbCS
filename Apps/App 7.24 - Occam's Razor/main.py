import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- 1. Page Config ---
st.set_page_config(
    page_title="Bayesian Model Comparison: Priors & Posteriors", layout="wide"
)

# --- 2. The Core Math (Gregory Sec 3.5) ---


def calculate_evidence(measured_val, sigma, prior_range):
    """
    Calculates the Evidence P(D|M) (Marginal Likelihood).
    """
    # --- Model 1: The "Sharp" Hypothesis (Exact 0) ---
    ev_m1 = norm.pdf(measured_val, loc=0, scale=sigma)

    # --- Model 2: The "Broad" Hypothesis (Uniform in [-I, I]) ---
    prior_density = 1.0 / (2 * prior_range)

    # We assume the Gaussian is contained within the prior range for this demo
    if -prior_range <= measured_val <= prior_range:
        ev_m2 = 1.0 * prior_density
    else:
        ev_m2 = 0.0

    # --- Occam Factor ---
    occam_factor = (sigma * np.sqrt(2 * np.pi)) / (2 * prior_range)
    max_likelihood_m2 = norm.pdf(measured_val, loc=measured_val, scale=sigma)

    return ev_m1, ev_m2, occam_factor, max_likelihood_m2


# --- 3. Visualization ---


def plot_distributions(measured_val, sigma, prior_range):
    """Visualizes the Parameter Space (mu)."""
    x = np.linspace(-prior_range * 1.2, prior_range * 1.2, 1000)
    likelihood = norm.pdf(x, loc=measured_val, scale=sigma)

    prior_m2 = np.zeros_like(x)
    mask = (x >= -prior_range) & (x <= prior_range)
    prior_m2[mask] = 1.0 / (2 * prior_range)

    fig = go.Figure()

    # Likelihood
    fig.add_trace(
        go.Scatter(
            x=x,
            y=likelihood,
            mode="lines",
            name="Likelihood $\\mathcal{L}(\\mu)$",
            fill="tozeroy",
            line=dict(color="#636EFA"),
            opacity=0.3,
        )
    )

    # Prior M2
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prior_m2,
            mode="lines",
            name="Prior $P(\\mu|M_2)$",
            line=dict(color="#EF553B", dash="dash"),
            hoverinfo="skip",
        )
    )

    # Model 1 marker
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers+text",
            name="Model $M_1$ Prediction",
            marker=dict(symbol="arrow-bar-up", size=20, color="black"),
            text=["M1"],
            textposition="top center",
        )
    )

    fig.update_layout(
        title="Parameter Space View",
        xaxis_title="Parameter Value ($\\mu$)",
        yaxis_title="Probability Density",
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_probability_shift(prior_m1, prior_m2, post_m1, post_m2):
    """Visualizes how the probability mass shifts from Prior to Posterior."""
    fig = go.Figure()

    # Prior Bar
    fig.add_trace(
        go.Bar(
            x=["Prior $P(M)$", "Posterior $P(M|D)$"],
            y=[prior_m1, post_m1],
            name="Model 1 (Simple)",
            marker_color="#119DFF",
        )
    )

    # Posterior Bar
    fig.add_trace(
        go.Bar(
            x=["Prior $P(M)$", "Posterior $P(M|D)$"],
            y=[prior_m2, post_m2],
            name="Model 2 (Complex)",
            marker_color="#EF553B",
        )
    )

    fig.update_layout(
        title="Belief Update",
        barmode="stack",
        yaxis_title="Probability",
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# --- 4. Main App ---


def main():
    st.title("Bayesian Model Comparison: Priors & Posteriors")
    st.markdown(
        "Compare a **Simple Model ($M_1$, $\\mu=0$)** vs a **Complex Model ($M_2$, $\\mu \\in [-I, I]$)**."
    )

    # --- Sidebar ---
    st.sidebar.header("1. Experimental Setup")
    measured_val = st.sidebar.number_input(
        "Measured Value ($D$)", -10.0, 10.0, 0.0, step=0.1
    )
    sigma = st.sidebar.number_input(
        "Measurement Noise ($\\sigma$)", 0.1, 5.0, 0.5, step=0.1
    )
    prior_range = st.sidebar.number_input(
        "M2 Prior Range ($I$)", 1.0, 100.0, 10.0, step=1.0
    )

    st.sidebar.divider()

    st.sidebar.header("2. Subjective Priors")
    st.sidebar.markdown("How much do you trust Model 1 before seeing data?")

    # We use one slider for M1, and derive M2 (since probabilities sum to 1)
    prob_m1 = st.sidebar.slider(
        "Prior Probability $P(M_1)$", 0.01, 0.99, 0.50, step=0.01
    )
    prob_m2 = 1.0 - prob_m1

    st.sidebar.write(f"**$P(M_1)={prob_m1:.2f}$** (Simple)")
    st.sidebar.write(f"**$P(M_2)={prob_m2:.2f}$** (Complex)")

    # --- Calculations ---

    # 1. Evidence P(D|M)
    ev_m1, ev_m2, occam_factor, max_like_m2 = calculate_evidence(
        measured_val, sigma, prior_range
    )

    # 2. Bayes Factor (Ratio of Evidences)
    # Avoid division by zero
    bayes_factor_12 = (ev_m1 / ev_m2) if ev_m2 > 1e-9 else 9999.0

    # 3. Posterior Odds
    # Odds_post = Odds_prior * Bayes_Factor
    prior_odds_12 = prob_m1 / prob_m2
    posterior_odds_12 = prior_odds_12 * bayes_factor_12

    # 4. Posterior Probabilities (Normalized)
    # P(M1|D) = (P(D|M1) * P(M1)) / P(D)
    # The denominator P(D) is the sum of the numerators of all models
    numerator_m1 = ev_m1 * prob_m1
    numerator_m2 = ev_m2 * prob_m2
    evidence_data = numerator_m1 + numerator_m2

    post_m1 = numerator_m1 / evidence_data
    post_m2 = numerator_m2 / evidence_data

    # --- Layout ---
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        # Visual 1: Parameter Space
        st.plotly_chart(
            plot_distributions(measured_val, sigma, prior_range),
            use_container_width=True,
        )

        # Visual 2: Probability Shift (Stacked Bar)
        st.plotly_chart(
            plot_probability_shift(prob_m1, prob_m2, post_m1, post_m2),
            use_container_width=True,
        )

    with col_right:
        st.subheader("The Verdict")

        # Evidence Card
        st.info(f"""
        **Evidence (Data Only)**
        * $P(D|M_1) = {ev_m1:.4f}$
        * $P(D|M_2) = {ev_m2:.4f}$
        
        **Bayes Factor ($B_{{12}}$):** {bayes_factor_12:.2f}
        *(Data favors M{1 if bayes_factor_12 > 1 else 2})*
        """)

        # Posterior Card
        result_color = "green" if post_m1 > post_m2 else "orange"
        winner = "Model 1" if post_m1 > post_m2 else "Model 2"

        st.markdown(
            f"""
        <div style="padding:15px; border-radius:5px; background-color:rgba(0,0,0,0.05); border-left: 5px solid {result_color}">
            <h3 style="margin:0">Winner: {winner}</h3>
            <p><strong>Posterior Probability: {max(post_m1, post_m2):.1%}</strong></p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("### The Equation")
        st.latex(r"""
        \underbrace{\frac{P(M_1|D)}{P(M_2|D)}}_{\text{Posterior Odds}} = 
        \underbrace{\frac{P(D|M_1)}{P(D|M_2)}}_{\text{Bayes Factor}} \times 
        \underbrace{\frac{P(M_1)}{P(M_2)}}_{\text{Prior Odds}}
        """)

        st.write("**Calculation:**")
        st.write(
            f"Posterior Odds = {bayes_factor_12:.2f} (Data) $\\times$ {prior_odds_12:.2f} (Belief)"
        )
        st.write(f"**Posterior Odds = {posterior_odds_12:.2f}**")


if __name__ == "__main__":
    main()
