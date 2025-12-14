import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import hypergeom
from dataclasses import dataclass, field
from typing import List

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Bayesian Inference: The Magician's Honest Trick", page_icon="🃏", layout="wide"
)

# --- Logic Core: Bayesian Model ---


@dataclass
class SimulationState:
    """Class to hold the state of the simulation."""

    true_red_count: int
    drawn_cards: List[str] = field(default_factory=list)
    # The probability distribution over the hypotheses H_0 to H_50
    # where H_k is the hypothesis that the deck has k red cards.
    posterior: np.ndarray = field(default_factory=lambda: np.zeros(51))
    prior: np.ndarray = field(default_factory=lambda: np.zeros(51))

    def __post_init__(self):
        # Initialize Prior if not already set (checking sum to see if empty)
        if np.sum(self.prior) == 0:
            self._initialize_prior()

    def _initialize_prior(self):
        """
        Calculates the initial prior based on the Magician's setup.
        Two decks (50R, 50B) mixed (100 total), then split into two 50-card decks.
        We care about the composition of ONE of those 50-card decks.
        This follows a Hypergeometric distribution.
        """
        M = 100  # Total population size (50R + 50B)
        n = 50  # Total number of Type I objects (Red cards in big pile)
        N = 50  # Number of draws (Size of the split deck)

        # k is the possible number of red cards in our deck (0 to 50)
        k_values = np.arange(0, 51)
        self.prior = hypergeom.pmf(k_values, M, n, N)
        self.posterior = self.prior.copy()

    def update(self, card_color: str):
        """
        Performs the Bayesian Update step.
        Posterior propto Likelihood * Prior
        """
        self.drawn_cards.append(card_color)

        # k represents the hypothesis: "The deck has k red cards"
        k_values = np.arange(0, 51)
        deck_size = 50

        if card_color == "Red":
            # Likelihood of drawing Red given k red cards in deck: P(R|H_k) = k / 50
            likelihood = k_values / deck_size
        else:
            # Likelihood of drawing Black given k red cards: P(B|H_k) = (50 - k) / 50
            likelihood = (deck_size - k_values) / deck_size

        # Unnormalized posterior
        unnormalized_posterior = likelihood * self.posterior

        # Normalize (Evidence)
        evidence = np.sum(unnormalized_posterior)

        # Avoid division by zero in edge cases (though unlikely here)
        if evidence > 0:
            self.posterior = unnormalized_posterior / evidence

    def reset(self):
        """Resets the simulation with a new random deck."""
        # Re-roll the true nature of the deck
        M, n, N = 100, 50, 50
        self.true_red_count = hypergeom.rvs(M, n, N)
        self.drawn_cards = []
        self._initialize_prior()


# --- Helper Functions ---


def draw_card_sim(true_red_count: int) -> str:
    """Simulates drawing a card with replacement from the true deck."""
    # Probability of drawing red from the ACTUAL deck
    p_red = true_red_count / 50.0
    return "Red" if np.random.random() < p_red else "Black"


def get_session_state() -> SimulationState:
    """Manages the SimulationState in Streamlit session_state."""
    # Check if we need to initialize or if the deck size changed (handling the update from 52 to 50)
    if (
        "sim_state" not in st.session_state
        or len(st.session_state.sim_state.prior) != 51
    ):
        # Initial random seed for the deck
        initial_red = hypergeom.rvs(100, 50, 50)
        st.session_state.sim_state = SimulationState(true_red_count=initial_red)
    return st.session_state.sim_state


# --- Visualization Components ---


def plot_distribution(state: SimulationState):
    """Plots the Prior vs Posterior distribution using Plotly."""
    k_values = np.arange(0, 51)
    proportions = k_values / 50.0

    fig = go.Figure()

    # Plot Prior (dashed line or light bar)
    fig.add_trace(
        go.Scatter(
            x=proportions,
            y=state.prior,
            mode="lines",
            name="Prior (Initial Belief)",
            line=dict(color="gray", dash="dash"),
            opacity=0.6,
        )
    )

    # Plot Posterior (Bar chart)
    # Color bars based on probability mass
    # Middle is 25 (proportion 0.5)
    colors = [
        "red" if p > 0.5 else "black" if p < 0.5 else "darkred" for p in proportions
    ]

    fig.add_trace(
        go.Bar(
            x=proportions,
            y=state.posterior,
            name="Posterior (Current Belief)",
            marker_color=colors,
            opacity=0.8,
        )
    )

    # Add a marker for the Maximum A Posteriori (MAP) estimate
    map_index = np.argmax(state.posterior)
    map_proportion = proportions[map_index]

    fig.add_vline(
        x=map_proportion,
        line_width=2,
        line_dash="dot",
        line_color="green",
        annotation_text="Most Likely",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Probability Distribution of Red Proportion",
        xaxis_title="Proportion of Red Cards (p)",
        yaxis_title="Probability P(p)",
        xaxis=dict(range=[0, 1], constrain="domain"),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def plot_history(state: SimulationState):
    """Plots the history of the expected value of Red Cards."""
    if not state.drawn_cards:
        return None

    # Reconstruct history of expectations (expensive but accurate)
    # Note: In a production app with huge data, we would cache this in state.
    # For a classroom demo (N < 100), recalculating is fine.

    temp_prior = state.prior.copy()
    k_values = np.arange(51)
    proportions = k_values / 50.0

    # Initial expectation (sum of p * prob(p))
    expectations = [np.sum(temp_prior * proportions)]

    current_post = temp_prior.copy()

    for card in state.drawn_cards:
        if card == "Red":
            likelihood = k_values / 50
        else:
            likelihood = (50 - k_values) / 50

        current_post = current_post * likelihood
        current_post /= np.sum(current_post)
        expectations.append(np.sum(current_post * proportions))

    x_axis = list(range(len(expectations)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis, y=expectations, mode="lines+markers", name="Expected Proportion"
        )
    )

    fig.update_layout(
        title="Convergence of Expectation",
        xaxis_title="Number of Draws",
        yaxis_title="Expected Proportion of Red Cards",
        yaxis=dict(range=[0, 1]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# --- Main App Interface ---


def main():
    sim = get_session_state()

    st.title("🃏 Bayesian Inference: The Magician's Deck")
    st.markdown("""
    **The Scenario:** A magician has two decks, removes 2 red and 2 black cards from the set, leaving **50 Red and 50 Black** cards total. 
    He mixes them, shuffles, and splits them into two equal piles of **50 cards**.
    One pile is chosen. You draw cards **with replacement** to figure out the **proportion** of Red cards in this specific pile.
    """)

    # --- Sidebar ---
    st.sidebar.header("Controls")

    if st.sidebar.button("Reset Simulation", type="primary"):
        sim.reset()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Stats")
    st.sidebar.metric("Cards Drawn", len(sim.drawn_cards))

    reds_drawn = sim.drawn_cards.count("Red")
    blacks_drawn = len(sim.drawn_cards) - reds_drawn

    col1, col2 = st.sidebar.columns(2)
    col1.metric("Red Drawn", reds_drawn)
    col2.metric("Black Drawn", blacks_drawn)

    reveal = st.sidebar.checkbox("Reveal True Deck (Instructor Mode)")
    if reveal:
        st.sidebar.success(f"True Red Count: {sim.true_red_count}")
        st.sidebar.info(f"True Proportion: {sim.true_red_count / 50:.2f}")

    # --- Interaction Area ---
    col_act, col_info = st.columns([1, 2])

    with col_act:
        st.subheader("Action")
        st.write("Draw a card from the deck:")

        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("Draw Card", width="stretch"):
            card = draw_card_sim(sim.true_red_count)
            sim.update(card)
            st.rerun()

        st.markdown("---")
        if len(sim.drawn_cards) > 0:
            last_card = sim.drawn_cards[-1]
            color = "red" if last_card == "Red" else "black"
            st.markdown(
                f"Last draw: :{'red' if color == 'red' else 'grey'}[**{last_card}**]"
            )

    with col_info:
        # Explanation of the math for students
        st.subheader("The Bayesian Update")
        with st.expander("See the Math", expanded=False):
            st.latex(r"""
            P(H_p | \text{Data}) = \frac{P(\text{Data} | H_p) \cdot P(H_p)}{P(\text{Data})}
            """)
            st.markdown("""
            * **Prior** $P(H_p)$: Determined by the Hypergeometric distribution (splitting the shuffled decks).
            * **Likelihood** $P(\text{Data}|H_p)$: Since we replace the card, if proportion is $p$, prob of drawing Red is $p$.
            """)

    # --- Main Visualizations ---

    # 1. Distribution Plot
    st.plotly_chart(plot_distribution(sim), width="stretch")

    # 2. History Plot (Only if data exists)
    if len(sim.drawn_cards) > 0:
        hist_fig = plot_history(sim)
        if hist_fig:
            st.plotly_chart(hist_fig, width="stretch")

    # 3. Interpretation
    st.markdown("### Interpretation")

    # Calculate Mean and Variance of Posterior
    k_values = np.arange(51)
    proportions = k_values / 50.0
    expected_prop = np.sum(sim.posterior * proportions)

    st.info(f"""
    Based on your observations, the model expects the proportion of Red cards is approximately **{expected_prop:.2f}**.
    
    The initial prior (before drawing) centered around **0.50** (because the split is random).
    As you draw more cards, the distribution becomes narrower (more confident) and shifts toward the true value.
    """)


if __name__ == "__main__":
    main()
