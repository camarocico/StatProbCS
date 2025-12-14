import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Model & Logic Classes
# -----------------------------------------------------------------------------


class BetaBinomialModel:
    """
    Represents the Bayesian model for a coin flip.
    Likelihood: Binomial
    Prior: Beta
    """

    def __init__(self, alpha_prior, beta_prior, n_trials, n_heads):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.n_trials = n_trials
        self.n_heads = n_heads
        self.n_tails = n_trials - n_heads

    def log_prior(self, theta):
        """Log-pdf of the Beta prior."""
        if 0 < theta < 1:
            return stats.beta.logpdf(theta, self.alpha_prior, self.beta_prior)
        return -np.inf

    def log_likelihood(self, theta):
        """Log-pmf of the Binomial likelihood."""
        # We only need the parts dependent on theta for MCMC, but using full logpmf is safer/clearer
        if 0 <= theta <= 1:
            return stats.binom.logpmf(self.n_heads, self.n_trials, theta)
        return -np.inf

    def log_posterior(self, theta):
        """Unnormalized Log-Posterior = Log-Likelihood + Log-Prior"""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return self.log_likelihood(theta) + lp

    def get_analytical_posterior_params(self):
        """Returns (alpha, beta) for the analytical posterior distribution."""
        # Conjugate update: alpha_post = alpha_prior + heads, beta_post = beta_prior + tails
        return (self.alpha_prior + self.n_heads, self.beta_prior + self.n_tails)


class MetropolisHastingsSampler:
    """
    Implements the Metropolis-Hastings MCMC algorithm.
    """

    def __init__(self, model, start_theta=0.5, step_size=0.1):
        self.model = model
        self.current_theta = start_theta
        self.step_size = (
            step_size  # Standard deviation of the proposal normal distribution
        )
        self.samples = []
        self.accepted = 0
        self.total_proposals = 0

    def run(self, n_iterations):
        """Runs the sampling loop."""

        # Pre-calculate current log posterior
        current_log_post = self.model.log_posterior(self.current_theta)

        for _ in range(n_iterations):
            # 1. Propose a new state: theta_new ~ N(theta_current, step_size)
            proposed_theta = np.random.normal(self.current_theta, self.step_size)

            # 2. Calculate acceptance probability
            # Note: Proposal is symmetric (Normal), so q(x|y) = q(y|x). They cancel out.
            # Ratio = P(theta_new|Data) / P(theta_current|Data)
            # In log space: Log_Ratio = Log_Post_New - Log_Post_Current

            proposed_log_post = self.model.log_posterior(proposed_theta)

            # If proposal is out of bounds (0,1), posterior is 0 (log is -inf), so we reject.
            if not np.isfinite(proposed_log_post):
                acceptance_prob = 0
            else:
                log_ratio = proposed_log_post - current_log_post
                # Cap at 0 (which is log(1)) because prob cannot be > 1
                acceptance_prob = np.exp(min(0, log_ratio))

            # 3. Accept or Reject
            if np.random.rand() < acceptance_prob:
                self.current_theta = proposed_theta
                current_log_post = proposed_log_post
                self.accepted += 1

            self.samples.append(self.current_theta)
            self.total_proposals += 1

        return np.array(self.samples)

    def get_acceptance_rate(self):
        return self.accepted / self.total_proposals if self.total_proposals > 0 else 0


# -----------------------------------------------------------------------------
# 2. Streamlit Application Logic
# -----------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="MCMC Interactive Demo", layout="wide")

    # --- CSS for better styling ---
    st.markdown(
        """
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #4B4B4B;}
    .sub-header {font-size: 1.5rem; color: #666;}
    .highlight {color: #1f77b4; font-weight: bold;}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header">MCMC for Bayesian Inference</div>',
        unsafe_allow_html=True,
    )
    st.markdown("Estimating coin bias $\\theta$ using Metropolis-Hastings sampling.")
    st.markdown("---")

    # --- Sidebar Controls ---
    st.sidebar.header("1. Simulation Settings")

    # Move configuration to an expander to hide "God Mode" settings
    with st.sidebar.expander("Configuration (God Mode)", expanded=False):
        st.markdown("Define the 'True World' and generate data.")
        true_theta = st.number_input(
            label="True Bias (Hidden)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="The actual probability of Heads.",
        )
        n_trials = st.number_input(
            label="Number of Flips (n)", value=10, min_value=1, step=1
        )
        # Manual Re-roll button
        if st.button("Flip Coin Again"):
            st.session_state.n_heads = np.random.binomial(n_trials, true_theta)

    show_true_bias = st.sidebar.checkbox("Reveal True Value", value=False)

    # --- Data Generation Logic (Session State) ---
    # Initialize state if not present
    if "n_heads" not in st.session_state:
        st.session_state.n_heads = np.random.binomial(n_trials, true_theta)
        st.session_state.last_sim_params = (n_trials, true_theta)

    # Check if simulation parameters changed since last run
    if st.session_state.last_sim_params != (n_trials, true_theta):
        st.session_state.n_heads = np.random.binomial(n_trials, true_theta)
        st.session_state.last_sim_params = (n_trials, true_theta)

    # Retrieve current data from state
    n_heads = st.session_state.n_heads

    # Display Observed Data in Sidebar
    st.sidebar.info(
        f"**Observed Data:**\n\nHeads: {n_heads}\n\nTails: {n_trials - n_heads}"
    )

    st.sidebar.header("2. Prior Beliefs (Beta)")
    st.sidebar.markdown("Prior: Beta($\\alpha, \\beta$)")
    alpha_prior = st.sidebar.number_input(
        label="Alpha ($\\alpha$)", min_value=0.1, max_value=100.0, value=1.0, step=0.1
    )
    beta_prior = st.sidebar.number_input(
        label="Beta ($\\beta$)", min_value=0.1, max_value=100.0, value=1.0, step=0.1
    )

    st.sidebar.header("3. MCMC Parameters")
    n_iterations = st.sidebar.number_input(
        label="Iterations", min_value=100, max_value=1000000, value=2000, step=100
    )
    burn_in = st.sidebar.number_input(
        label="Burn-in Period",
        min_value=0,
        max_value=n_iterations // 2,
        value=200,
        step=50,
    )
    step_size = st.sidebar.number_input(
        label="Proposal Step Size (Std Dev)",
        min_value=0.01,
        max_value=1.0,
        value=0.2,
        step=0.01,
    )

    start_theta = st.sidebar.number_input(
        label="Starting $\\theta$", min_value=0.1, max_value=0.9, value=0.5, step=0.1
    )

    # --- Run Simulation ---
    # Instantiate Model
    model = BetaBinomialModel(alpha_prior, beta_prior, n_trials, n_heads)

    # Analytical Solution for comparison
    true_alpha, true_beta = model.get_analytical_posterior_params()
    analytical_mean = true_alpha / (true_alpha + true_beta)

    # Run MCMC
    sampler = MetropolisHastingsSampler(
        model, start_theta=start_theta, step_size=step_size
    )

    # Run sampler
    samples = sampler.run(n_iterations)

    # Process Samples
    burnt_samples = samples[burn_in:]
    mcmc_mean = np.mean(burnt_samples)
    acceptance_rate = sampler.get_acceptance_rate()

    # --- Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Trace Plot")
        st.markdown("This shows the path the sampler took through the parameter space.")

        trace_fig = go.Figure()

        # Plot Burn-in
        trace_fig.add_trace(
            go.Scatter(
                x=list(range(burn_in)),
                y=samples[:burn_in],
                mode="lines",
                name="Burn-in",
                line=dict(color="red", width=1),
                opacity=0.5,
            )
        )

        # Plot Valid Samples
        trace_fig.add_trace(
            go.Scatter(
                x=list(range(burn_in, n_iterations)),
                y=burnt_samples,
                mode="lines",
                name="Posterior Samples",
                line=dict(color="#1f77b4", width=1),
            )
        )

        trace_fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Theta (Bias)",
            yaxis_range=[0, 1],
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=0, b=0),
            height=350,
        )
        st.plotly_chart(trace_fig, width="stretch")

        st.subheader("Posterior Distribution")
        st.markdown("Comparing MCMC samples (Histogram) vs. Analytical Solution.")

        hist_fig = go.Figure()

        # MCMC Histogram
        hist_fig.add_trace(
            go.Histogram(
                x=burnt_samples,
                histnorm="probability density",
                name="MCMC Samples",
                marker_color="#1f77b4",
                opacity=0.6,
                xbins=dict(start=0, end=1, size=0.01),
            )
        )

        # Analytical Curve
        x_axis = np.linspace(0, 1, 500)
        y_axis = stats.beta.pdf(x_axis, true_alpha, true_beta)

        hist_fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_axis,
                mode="lines",
                name=f"True Posterior Beta({true_alpha}, {true_beta})",
                line=dict(color="red", width=3),
            )
        )

        # Add True Bias Line (Conditional)
        if show_true_bias:
            hist_fig.add_vline(
                x=true_theta,
                line_width=2,
                line_dash="dash",
                line_color="green",
                annotation_text="True Bias",
            )

        hist_fig.update_layout(
            xaxis_title="Theta (Bias)",
            yaxis_title="Density",
            xaxis_range=[0, 1],
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=0, b=0),
            height=350,
        )
        st.plotly_chart(hist_fig, width="stretch")

    with col2:
        st.markdown("### Diagnostics & Results")

        error_val = abs(mcmc_mean - analytical_mean)  # Comparing MCMC to Analytical
        error_val_true = abs(mcmc_mean - true_theta)  # Comparing MCMC to True
        error_color = "green" if error_val < 0.02 else "red"
        error_color_true = "green" if error_val_true < 0.02 else "red"
        error_html = f"<p style='color: {error_color}'>MCMC vs Analytical Error: {error_val:.4f}</p>"

        # Logic for hiding/showing results
        if show_true_bias:
            true_bias_display = f"{true_theta:.4f}"
            error_html_true = f"<p style='color: {error_color_true}'>MCMC vs True Bias Error: {error_val_true:.4f}</p>"
        else:
            true_bias_display = "???"
            error_html_true = "<p>MCMC vs True Bias Error: <i>Hidden</i></p>"

            # Metric Cards
        st.markdown(
            f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h4>Estimation Results</h4>
            <p><b>True Bias (Hidden):</b> {true_bias_display}</p>
            <p><b>Analytical Mean:</b> {analytical_mean:.4f}</p>
            <p><b>MCMC Mean:</b> {mcmc_mean:.4f}</p>
            {error_html}
            {error_html_true}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
            <h4>Sampler Health</h4>
            <p><b>Acceptance Rate:</b> {acceptance_rate:.2%}</p>
            <small>Target is usually between 20% - 50% for high dimensions, but for 1D ~40-60% is good.</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### Learning Points")
        st.info("""
        1. **Simulation**: We generated coin flips using a 'True Bias' (Green dashed line). 
        2. **Sampling Error**: Even with a known True Bias, the *observed* heads/tails (Data) might not perfectly match it, especially with small `n`.
        3. **Posterior**: The posterior pulls the prior towards the *observed data*, not necessarily the *true bias* directly.
        """)

        # Data Summary
        st.markdown("### Data Summary")
        st.write(
            pd.DataFrame(
                {
                    "Metric": ["Heads", "Tails", "Total"],
                    "Count": [n_heads, n_trials - n_heads, n_trials],
                }
            ).set_index("Metric")
        )


if __name__ == "__main__":
    main()
