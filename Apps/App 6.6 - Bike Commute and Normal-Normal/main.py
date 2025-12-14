import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Normal-Normal Conjugacy: Bike Commute", page_icon="🚲", layout="wide"
)

# --- Logic Core: Bayesian Model (Normal-Normal) ---


@dataclass
class NormalSimulationState:
    """
    Holds the state for a Normal-Normal conjugate simulation.
    We are estimating the Mean (mu) of the commute time.
    We assume the variability of the commute (sigma_likelihood) is known/fixed for simplicity.
    """

    # True Reality (Hidden from student initially)
    true_mu: float
    true_sigma: float

    # Current Belief (Hyperparameters for the Normal distribution of mu)
    # Moved up because they are non-default arguments
    current_mu_belief: float  # Mean of the posterior
    current_sigma_belief: float  # Std Dev of the posterior (uncertainty)

    # History
    measured_times: List[float] = field(default_factory=list)

    # Initial Belief (for plotting comparison)
    prior_mu_belief: float = 15.0
    prior_sigma_belief: float = 5.0

    def update(self, measured_time: float, likelihood_sigma: float):
        """
        Performs the Bayesian Update for a Normal Likelihood with known variance
        and a Normal Prior.

        Math:
        Precision_new = Precision_old + Precision_likelihood
        Mean_new = (Precision_old * Mean_old + Precision_likelihood * Data) / Precision_new
        """
        self.measured_times.append(measured_time)

        # Convert standard deviations to precisions (tau = 1/sigma^2)
        tau_belief = 1.0 / (self.current_sigma_belief**2)
        tau_data = 1.0 / (likelihood_sigma**2)

        # Calculate new precision
        tau_new = tau_belief + tau_data

        # Calculate new mean
        # weighted average of prior mean and data value, weighted by their precisions
        mu_new = (
            tau_belief * self.current_mu_belief + tau_data * measured_time
        ) / tau_new

        # Convert back to standard deviation
        sigma_new = np.sqrt(1.0 / tau_new)

        # Update state
        self.current_mu_belief = mu_new
        self.current_sigma_belief = sigma_new

    def reset(
        self, initial_mu: float, initial_sigma: float, true_mu: float, true_sigma: float
    ):
        """Resets the simulation with specific parameters."""
        self.current_mu_belief = initial_mu
        self.current_sigma_belief = initial_sigma
        self.prior_mu_belief = initial_mu
        self.prior_sigma_belief = initial_sigma

        self.true_mu = true_mu
        self.true_sigma = true_sigma
        self.measured_times = []


# --- Helper Functions ---


def get_session_state() -> NormalSimulationState:
    """Manages the SimulationState in Streamlit session_state."""
    if "normal_sim_state" not in st.session_state:
        # Default initialization
        st.session_state.normal_sim_state = NormalSimulationState(
            true_mu=18.0,
            true_sigma=3.0,
            current_mu_belief=15.0,
            current_sigma_belief=5.0,
        )
    return st.session_state.normal_sim_state


# --- Visualization Components ---


def plot_distributions(state: NormalSimulationState, show_truth: bool = False):
    """Plots the Prior and Posterior Normal distributions."""

    # Determine x-axis range based on the distributions to ensure they fit
    # We look at the true mean, prior mean, and current mean +/- 4 sigmas
    min_x = min(
        state.true_mu - 3 * state.true_sigma,
        state.prior_mu_belief - 3 * state.prior_sigma_belief,
        5,
    )
    max_x = max(
        state.true_mu + 3 * state.true_sigma,
        state.prior_mu_belief + 3 * state.prior_sigma_belief,
        35,
    )

    x_axis = np.linspace(min_x, max_x, 500)

    # Calculate PDFs
    # 1. Prior
    prior_pdf = norm.pdf(x_axis, state.prior_mu_belief, state.prior_sigma_belief)

    # 2. Posterior
    post_pdf = norm.pdf(x_axis, state.current_mu_belief, state.current_sigma_belief)

    fig = go.Figure()

    # Add Prior
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=prior_pdf,
            mode="lines",
            name="Prior Belief",
            line=dict(color="gray", dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(128, 128, 128, 0.1)",
        )
    )

    # Add Posterior
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=post_pdf,
            mode="lines",
            name="Posterior Belief",
            line=dict(color="#636EFA", width=3),
            fill="tozeroy",
            fillcolor="rgba(99, 110, 250, 0.2)",
        )
    )

    # Add True Value Marker (Vertical Line)
    if show_truth:
        fig.add_vline(
            x=state.true_mu,
            line_width=2,
            line_dash="solid",
            line_color="green",
            annotation_text="Truth",
            annotation_position="top right",
        )

    # Add Last Data Point marker if exists
    if state.measured_times:
        last_val = state.measured_times[-1]
        fig.add_vline(
            x=last_val,
            line_width=1,
            line_dash="dot",
            line_color="red",
            opacity=0.5,
            annotation_text="Last Obs",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title="Belief about Average Commute Time (μ)",
        xaxis_title="Time (minutes)",
        yaxis_title="Probability Density",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def plot_convergence(state: NormalSimulationState, show_truth: bool = False):
    """Plots the history of the estimated mean and confidence interval."""
    if not state.measured_times:
        return None

    # Reconstruct history for plotting
    # We need to re-run the logic temporarily to get the path
    # (Again, efficient enough for N < 100 in a demo)

    mus = [state.prior_mu_belief]
    sigmas = [state.prior_sigma_belief]

    curr_mu = state.prior_mu_belief
    curr_sigma = state.prior_sigma_belief

    # The likelihood sigma is fixed in reality (state.true_sigma),
    # but in the model, we often treat it as a known parameter.
    # We will use the true_sigma as the "known" traffic variability for the update.
    likelihood_sigma = state.true_sigma

    for x in state.measured_times:
        tau_curr = 1.0 / (curr_sigma**2)
        tau_data = 1.0 / (likelihood_sigma**2)
        tau_new = tau_curr + tau_data

        curr_mu = (tau_curr * curr_mu + tau_data * x) / tau_new
        curr_sigma = np.sqrt(1.0 / tau_new)

        mus.append(curr_mu)
        sigmas.append(curr_sigma)

    x_axis = list(range(len(mus)))
    upper_bound = [m + 1.96 * s for m, s in zip(mus, sigmas)]
    lower_bound = [m - 1.96 * s for m, s in zip(mus, sigmas)]

    fig = go.Figure()

    # 95% CI Area
    fig.add_trace(
        go.Scatter(
            x=x_axis + x_axis[::-1],  # x, then x reversed
            y=upper_bound + lower_bound[::-1],  # upper, then lower reversed
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Credible Interval",
        )
    )

    # Mean Line
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=mus,
            mode="lines+markers",
            name="Estimated Mean",
            line=dict(color="#636EFA"),
        )
    )

    # True Mean Line
    if show_truth:
        fig.add_trace(
            go.Scatter(
                x=[0, len(mus) - 1],
                y=[state.true_mu, state.true_mu],
                mode="lines",
                name="True Mean",
                line=dict(color="green", dash="dot"),
            )
        )

    fig.update_layout(
        title="Convergence of Estimate",
        xaxis_title="Number of Days Measured",
        yaxis_title="Estimated Commute Time (min)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# --- Main App Interface ---


def main():
    sim = get_session_state()

    st.title("🚲 Normal-Normal Conjugacy: Bike Commute")
    st.markdown("""
    **The Scenario:** You want to know your average bike commute time (μ). 
    You have a rough guess, but you are not sure. 
    Every day, you measure how long the ride takes. Conditions (wind, traffic lights) vary day-to-day (Standard Deviation σ), but we assume this variability is roughly known.
    """)

    # --- Sidebar Configuration ---
    st.sidebar.header("1. Your Prior Beliefs")

    # Using sliders with keys to avoid state conflicts, but we need to manually sync them with reset
    prior_mu_input = st.sidebar.slider(
        "I think my average is (minutes):",
        min_value=5.0,
        max_value=60.0,
        value=15.0,
        step=0.5,
        help="This is the Mean of your Prior distribution (μ_0).",
    )

    prior_sigma_input = st.sidebar.slider(
        "I am this uncertain (+/- minutes):",
        min_value=0.5,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="This is the Standard Deviation of your Prior (σ_0). Larger = less sure.",
    )

    true_sigma_input = st.sidebar.slider(
        "Ride Variability (Daily Std Dev):",
        min_value=1.0,
        max_value=15.0,
        value=3.0,
        step=0.5,
        help="How much the ride varies day-to-day due to wind or red lights.",
    )

    st.sidebar.markdown("---")

    # Moved True Reality to Expander to hide it initially
    with st.sidebar.expander("2. The True Reality (Instructor)", expanded=False):
        st.markdown("*Configure the actual riding conditions.*")

        true_mu_input = st.slider(
            "Actual Average Time:", min_value=5.0, max_value=60.0, value=20.0, step=0.5
        )

        show_truth = st.checkbox("Show True Mean on Plots", value=False)

    # Reset Logic
    # We check if the settings changed significantly or if it's the first run
    # To keep it simple, we just have a hard Reset button
    if st.sidebar.button("Start / Reset Simulation", type="primary"):
        sim.reset(
            initial_mu=prior_mu_input,
            initial_sigma=prior_sigma_input,
            true_mu=true_mu_input,
            true_sigma=true_sigma_input,
        )
        st.rerun()

    # --- Main Area ---

    col_act, col_info = st.columns([1, 2])

    with col_act:
        st.subheader("Action")
        st.write("Simulate one day of commuting:")

        if st.button("Measure Commute", width="stretch"):
            # Generate data point based on TRUTH
            # Data ~ N(true_mu, true_sigma)
            obs = np.random.normal(sim.true_mu, sim.true_sigma)

            # Update belief based on observed data
            # IMPORTANT: In the update step, we need to know the likelihood sigma.
            # In a textbook example, sigma is often "known". We use true_sigma.
            sim.update(obs, sim.true_sigma)
            st.rerun()

        if sim.measured_times:
            last_val = sim.measured_times[-1]
            st.metric("Today's Commute", f"{last_val:.1f} min")
            st.markdown(f"**Total Days:** {len(sim.measured_times)}")

    with col_info:
        st.subheader("The Normal-Normal Update")
        with st.expander("See the Math", expanded=False):
            st.markdown(r"""
            This models a **Normal Prior** with a **Normal Likelihood** (variance known).
            
            We update estimations using **Precision** ($\tau = 1/\sigma^2$).
            
            $$
            \tau_{new} = \tau_{old} + \tau_{data}
            $$
            
            The new mean is a weighted average:
            $$
            \mu_{new} = \frac{\tau_{old}\mu_{old} + \tau_{data}x}{\tau_{new}}
            $$
            """)
            st.info(
                "Notice: If your Prior is very uncertain (low precision), the Data (high precision) pulls the estimate quickly."
            )

    # --- Visualizations ---

    st.plotly_chart(plot_distributions(sim, show_truth), width="stretch")

    if sim.measured_times:
        st.plotly_chart(plot_convergence(sim, show_truth), width="stretch")

    # --- Statistics Table ---
    if sim.measured_times:
        st.markdown("### Current Belief Stats")
        cols = st.columns(3)
        cols[0].metric("Estimated Mean", f"{sim.current_mu_belief:.2f}")
        cols[1].metric("Uncertainty (Std Dev)", f"{sim.current_sigma_belief:.2f}")
        # 95% Credible Interval
        lower = sim.current_mu_belief - 1.96 * sim.current_sigma_belief
        upper = sim.current_mu_belief + 1.96 * sim.current_sigma_belief
        cols[2].metric("95% Credible Interval", f"[{lower:.1f}, {upper:.1f}]")


if __name__ == "__main__":
    main()
