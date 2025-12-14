import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gamma
from dataclasses import dataclass, field
from typing import List

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Gamma-Poisson Conjugacy: Daily Emails", page_icon="📧", layout="wide"
)

# --- Logic Core: Bayesian Model (Gamma-Poisson) ---


@dataclass
class GammaSimulationState:
    """
    Holds the state for a Gamma-Poisson conjugate simulation.
    We are estimating the Rate (lambda) of emails per day.

    Likelihood: Poisson(lambda)
    Prior: Gamma(alpha, beta)
    """

    # True Reality (Hidden from student initially)
    true_lambda: float

    # Current Belief (Hyperparameters for the Gamma distribution)
    current_alpha: float  # Shape parameter (roughly "total emails seen")
    current_beta: float  # Rate parameter (roughly "total days observed")

    # History
    daily_counts: List[int] = field(default_factory=list)

    # Initial Belief (for plotting comparison)
    prior_alpha: float = 4.0
    prior_beta: float = 2.0

    def update(self, daily_count: int):
        """
        Performs the Bayesian Update for a Poisson Likelihood and Gamma Prior.

        Math:
        alpha_new = alpha_old + data_count
        beta_new = beta_old + 1 (representing 1 unit of time/day)
        """
        self.daily_counts.append(daily_count)

        # Bayesian Update
        self.current_alpha += daily_count
        self.current_beta += 1.0

    def reset(self, initial_mean: float, initial_var: float, true_lambda: float):
        """
        Resets the simulation.
        Converts Mean/Variance inputs into Alpha/Beta for the Gamma distribution.

        Mean = alpha / beta
        Var = alpha / beta^2
        => beta = Mean / Var
        => alpha = Mean * beta
        """
        # Calculate alpha/beta from mean/variance
        # Avoid division by zero
        if initial_var < 0.01:
            initial_var = 0.01

        beta = initial_mean / initial_var
        alpha = initial_mean * beta

        self.current_alpha = alpha
        self.current_beta = beta
        self.prior_alpha = alpha
        self.prior_beta = beta

        self.true_lambda = true_lambda
        self.daily_counts = []


# --- Helper Functions ---


def get_session_state() -> GammaSimulationState:
    """Manages the SimulationState in Streamlit session_state."""
    if "gamma_sim_state" not in st.session_state:
        # Default initialization based on prompt: Mean=2, Var=1
        # beta = 2/1 = 2, alpha = 2*2 = 4
        st.session_state.gamma_sim_state = GammaSimulationState(
            true_lambda=3.5,  # Let's make the truth a bit different from the guess
            current_alpha=4.0,
            current_beta=2.0,
            prior_alpha=4.0,
            prior_beta=2.0,
        )
    return st.session_state.gamma_sim_state


# --- Visualization Components ---


def plot_distributions(state: GammaSimulationState, show_truth: bool = False):
    """Plots the Prior and Posterior Gamma distributions."""

    # Determine x-axis range
    # Gamma support is [0, infinity). We look at reasonable range around means.
    prior_mean = state.prior_alpha / state.prior_beta
    curr_mean = state.current_alpha / state.current_beta

    # We also need to consider the data range for the x-axis now that we plot data
    max_data = max(state.daily_counts) if state.daily_counts else 0

    max_x = max(
        state.true_lambda * 2.5, prior_mean * 3, curr_mean * 3, max_data + 2, 10
    )
    x_axis = np.linspace(0, max_x, 500)

    # Calculate PDFs (scale = 1/beta)
    # 1. Prior
    prior_pdf = gamma.pdf(x_axis, a=state.prior_alpha, scale=1.0 / state.prior_beta)

    # 2. Posterior
    post_pdf = gamma.pdf(x_axis, a=state.current_alpha, scale=1.0 / state.current_beta)

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
            line=dict(color="#00CC96", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 204, 150, 0.2)",
        )
    )

    # Add True Value Marker (Vertical Line)
    if show_truth:
        fig.add_vline(
            x=state.true_lambda,
            line_width=2,
            line_dash="solid",
            line_color="#EF553B",
            annotation_text="Truth (λ)",
            annotation_position="top right",
        )

    # Add Data Markers (Observed Daily Counts)
    if state.daily_counts:
        # Historical data (thin, transparent)
        # We loop through all but the last one
        for x in state.daily_counts[:-1]:
            fig.add_vline(x=x, line_width=1, line_color="#AB63FA", opacity=0.15)

        # Latest data (thicker, dashed, labeled)
        last_count = state.daily_counts[-1]
        fig.add_vline(
            x=last_count,
            line_width=2,
            line_dash="dot",
            line_color="#AB63FA",
            annotation_text=f"Data: {last_count}",
            annotation_position="top left",
        )

        # Add a dummy trace for the legend so "Observed Data" appears
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="#AB63FA", dash="dot", width=2),
                name="Observed Data",
            )
        )

    fig.update_layout(
        title="Belief about Daily Email Rate (λ)",
        xaxis_title="Average Emails per Day (λ)",
        yaxis_title="Probability Density",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def plot_convergence(state: GammaSimulationState, show_truth: bool = False):
    """Plots the history of the estimated mean and credible interval."""
    if not state.daily_counts:
        return None

    # Reconstruct history for plotting
    mus = [state.prior_alpha / state.prior_beta]

    # 95% Interval for Gamma is not symmetric, use ppf
    lowers = [gamma.ppf(0.025, a=state.prior_alpha, scale=1.0 / state.prior_beta)]
    uppers = [gamma.ppf(0.975, a=state.prior_alpha, scale=1.0 / state.prior_beta)]

    curr_alpha = state.prior_alpha
    curr_beta = state.prior_beta

    for x in state.daily_counts:
        curr_alpha += x
        curr_beta += 1

        mus.append(curr_alpha / curr_beta)
        lowers.append(gamma.ppf(0.025, a=curr_alpha, scale=1.0 / curr_beta))
        uppers.append(gamma.ppf(0.975, a=curr_alpha, scale=1.0 / curr_beta))

    x_axis = list(range(len(mus)))

    fig = go.Figure()

    # 95% CI Area
    fig.add_trace(
        go.Scatter(
            x=x_axis + x_axis[::-1],  # x, then x reversed
            y=uppers + lowers[::-1],  # upper, then lower reversed
            fill="toself",
            fillcolor="rgba(0, 204, 150, 0.2)",
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
            name="Estimated Rate",
            line=dict(color="#00CC96"),
        )
    )

    # True Mean Line
    if show_truth:
        fig.add_trace(
            go.Scatter(
                x=[0, len(mus) - 1],
                y=[state.true_lambda, state.true_lambda],
                mode="lines",
                name="True Rate",
                line=dict(color="#EF553B", dash="dot"),
            )
        )

    fig.update_layout(
        title="Convergence of Estimate",
        xaxis_title="Number of Days Observed",
        yaxis_title="Estimated Rate (Emails/Day)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_daily_data(state: GammaSimulationState):
    """Plots the raw observed data (counts per day)."""
    if not state.daily_counts:
        return None

    # X-axis: Day 1, Day 2, etc.
    days = list(range(1, len(state.daily_counts) + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=days, y=state.daily_counts, name="Daily Count", marker_color="#AB63FA")
    )

    fig.update_layout(
        title="Observed Data: Daily Email Counts",
        xaxis_title="Day",
        yaxis_title="Number of Emails Received",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        bargap=0.2,
    )
    return fig


# --- Main App Interface ---


def main():
    sim = get_session_state()

    st.title("📧 Bayesian Inference: Email Counts")
    st.markdown("""
    **The Scenario:** You want to estimate the average number of emails you get per day (λ).
    You count the number of emails in your inbox at the end of every day.
    
    Since the data is a **count** (Poisson distribution), we use the **Gamma** distribution as our Prior.
    """)

    # --- Sidebar Configuration ---
    st.sidebar.header("1. Your Prior Beliefs")

    # User inputs Mean and Variance -> Convert to Alpha/Beta
    prior_mean_input = st.sidebar.slider(
        "I think I get this many emails/day (Mean):",
        min_value=1.0,
        max_value=100.0,
        value=21.0,
        step=1.0,
        help="This sets the center of your Prior.",
    )

    prior_var_input = st.sidebar.slider(
        "Uncertainty (Variance):",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Higher variance means you are less sure. This affects the width of the curve.",
    )

    # Calculate alpha/beta just for display
    # beta = mean/var, alpha = mean*beta
    display_beta = prior_mean_input / prior_var_input if prior_var_input > 0 else 0
    display_alpha = prior_mean_input * display_beta
    st.sidebar.caption(f"Implies Gamma(α={display_alpha:.1f}, β={display_beta:.1f})")

    st.sidebar.markdown("---")

    with st.sidebar.expander("2. The True Reality (Instructor)", expanded=False):
        st.markdown("*Configure the actual email rate.*")

        true_lambda_input = st.slider(
            "Actual Average Rate (λ):",
            min_value=0.5,
            max_value=20.0,
            value=3.5,
            step=0.5,
        )

        show_truth = st.checkbox("Show True Rate on Plots", value=False)

    # Reset Logic
    if st.sidebar.button("Start / Reset Simulation", type="primary"):
        sim.reset(
            initial_mean=prior_mean_input,
            initial_var=prior_var_input,
            true_lambda=true_lambda_input,
        )
        st.rerun()

    # --- Main Area ---

    col_act, col_info = st.columns([1, 2])

    with col_act:
        st.subheader("Action")
        st.write("Check inbox for one day:")

        if st.button("Simulate One Day", width="stretch"):
            # Generate data point based on TRUTH
            # Data ~ Poisson(true_lambda)
            obs = np.random.poisson(sim.true_lambda)

            # Update belief
            sim.update(obs)
            st.rerun()

        if sim.daily_counts:
            last_val = sim.daily_counts[-1]
            st.metric("Emails Today", f"{last_val}")
            st.markdown(f"**Days Observed:** {len(sim.daily_counts)}")

    with col_info:
        st.subheader("The Gamma-Poisson Update")
        with st.expander("See the Math", expanded=False):
            st.markdown(r"""
            **Likelihood:** $Data \sim \text{Poisson}(\lambda)$
            
            **Prior:** $\lambda \sim \text{Gamma}(\alpha, \beta)$
            
            **Update Rule:**
            Add total count of emails to $\alpha$.
            Add total days passed to $\beta$.
            
            $$
            \alpha_{new} = \alpha_{old} + x
            $$
            $$
            \beta_{new} = \beta_{old} + 1
            $$
            
            The Expected Value is $E[\lambda] = \alpha / \beta$.
            """)
            st.info(
                "The Gamma distribution handles rates that must be positive (unlike the Normal, which allows negatives)."
            )

    # --- Visualizations ---

    st.plotly_chart(plot_distributions(sim, show_truth), width="stretch")

    if sim.daily_counts:
        col_hist, col_data = st.columns(2)

        with col_hist:
            st.plotly_chart(plot_convergence(sim, show_truth), width="stretch")

        with col_data:
            st.plotly_chart(plot_daily_data(sim), width="stretch")

    # --- Statistics Table ---
    st.markdown("### Current Belief Stats")

    # Calculate stats
    curr_mean = sim.current_alpha / sim.current_beta
    curr_var = sim.current_alpha / (sim.current_beta**2)
    # Mode is (alpha-1)/beta for alpha >= 1
    curr_mode = (
        (sim.current_alpha - 1) / sim.current_beta if sim.current_alpha >= 1 else 0
    )

    cols = st.columns(4)
    cols[0].metric("Estimated Rate", f"{curr_mean:.2f}")
    cols[1].metric("Most Likely Rate (Mode)", f"{curr_mode:.2f}")
    cols[2].metric("Uncertainty (Var)", f"{curr_var:.2f}")
    cols[3].metric("Effective Sample Size", f"{sim.current_beta:.1f} days")


if __name__ == "__main__":
    main()
