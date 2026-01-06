import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Bayesian Inference: Opinion Dynamics", page_icon="📉", layout="wide"
)

# --- Data Structures & Logic ---


@dataclass
class AgentConfig:
    name: str
    color: str  # Added color for consistent visualization
    initial_prior_safe: float
    p_claim_given_safe: float
    p_claim_given_unsafe: float
    description: str


class BayesianAgent:
    def __init__(self, config: AgentConfig):
        self.name = config.name
        self.color = config.color
        self.prior_safe = config.initial_prior_safe
        self.prior_unsafe = 1.0 - config.initial_prior_safe
        self.p_d_given_s = config.p_claim_given_safe
        self.p_d_given_not_s = config.p_claim_given_unsafe
        self.description = config.description

        self.posterior_safe = self.calculate_posterior(self.prior_safe)

    def calculate_posterior(self, prior_s: float) -> float:
        """
        Calculates P(Safe | Claim Unsafe) for any given Prior P(S).
        Useful for plotting the entire belief curve.
        """
        prior_not_s = 1.0 - prior_s
        likelihood = self.p_d_given_s
        evidence_marginal = (self.p_d_given_s * prior_s) + (
            self.p_d_given_not_s * prior_not_s
        )

        if evidence_marginal == 0:
            return 0.0

        return (likelihood * prior_s) / evidence_marginal


# --- Visualization ---


def create_bar_chart(agents: List[BayesianAgent]) -> go.Figure:
    """Standard Prior vs Posterior Bar Chart."""
    names = [a.name for a in agents]
    priors = [a.prior_safe for a in agents]
    posteriors = [a.posterior_safe for a in agents]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names,
            y=priors,
            name="Prior Belief",
            marker_color="#cbd5e1",
            text=[f"{p:.2f}" for p in priors],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=names,
            y=posteriors,
            name="Posterior Belief",
            marker_color=[a.color for a in agents],
            text=[f"{p:.2f}" for p in posteriors],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Current Belief State",
        yaxis_title="Probability of Safety",
        barmode="group",
        yaxis=dict(range=[0, 1.1]),
        template="plotly_white",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def create_belief_landscape(agents: List[BayesianAgent]) -> go.Figure:
    """
    Plots the 'Transfer Function' of each agent.
    X-axis: Input Prior
    Y-axis: Output Posterior
    """
    x_range = np.linspace(0, 1, 100)
    fig = go.Figure()

    # 1. Add Diagonal Reference (No Update Line)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="No Change (Reference)",
            line=dict(color="gray", dash="dot", width=1),
            hoverinfo="skip",
        )
    )

    # 2. Add Curves and Current State Markers for each agent
    for agent in agents:
        # Calculate the curve for this agent's "Personality"
        y_values = [agent.calculate_posterior(x) for x in x_range]

        # Plot the Curve
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_values,
                mode="lines",
                name=f"{agent.name} Curve",
                line=dict(color=agent.color, width=2),
                opacity=0.6,
                hoverinfo="skip",  # Cleaner hover
            )
        )

        # Plot the Actual Current Position (The Dot)
        fig.add_trace(
            go.Scatter(
                x=[agent.prior_safe],
                y=[agent.posterior_safe],
                mode="markers",
                name=f"{agent.name} Actual",
                marker=dict(
                    color=agent.color, size=14, line=dict(width=2, color="white")
                ),
                hovertemplate=f"<b>{agent.name}</b><br>Prior: %{{x:.2f}}<br>Posterior: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Opinion Dynamics: The Belief Landscape",
        xaxis_title="Input: Prior Belief P(S)",
        yaxis_title="Output: Posterior Belief P(S|D)",
        template="plotly_white",
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 1.0]),
        height=500,
        legend=dict(x=0.02, y=0.98),
    )

    # Annotations to explain regions
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text="Unchanged Beliefs",
        showarrow=False,
        font=dict(size=10, color="gray"),
        textangle=-45,
    )

    return fig


# --- Main Application ---


def main():
    st.title("Bayesian Inference: How Opinions Can Become Polarized")

    col_intro, col_math = st.columns([2, 1])
    with col_intro:
        st.markdown("""
        **The Scenario:** Mr. N claims a drug is **unsafe**. 
        
        This dashboard visualizes how different people (Agents) update their beliefs. 
        Notice how **Agent A** and **Agent C** can "switch" places in confidence levels, 
        even if they start with similar beliefs, purely because they interpret Mr. N's motives differently.
        """)
    with col_math:
        st.latex(r"""P(S|D) = \frac{P(D|S)P(S)}{P(D)}""")

    st.divider()

    # --- Sidebar Controls ---
    st.sidebar.header("Agent Configuration")

    # Defaults based on Jaynes Section 5.3
    with st.sidebar.expander("Agent A (The Disciple)", expanded=True):
        st.caption("Trusts Mr. N implicitly.")
        prior_a = st.slider("A: Prior P(S)", 0.0, 1.0, 0.9)
        la_fa = st.slider("A: P(Claim|Safe)", 0.0, 1.0, 0.01)
        la_td = st.slider("A: P(Claim|Unsafe)", 0.0, 1.0, 1.0)

    with st.sidebar.expander("Agent B (The Skeptic)", expanded=False):
        st.caption("Thinks Mr. N is erratic.")
        prior_b = st.slider("B: Prior P(S)", 0.0, 1.0, 0.1)
        lb_fa = st.slider("B: P(Claim|Safe)", 0.0, 1.0, 0.4)
        lb_td = st.slider("B: P(Claim|Unsafe)", 0.0, 1.0, 0.6)

    with st.sidebar.expander("Agent C (The Cynic)", expanded=False):
        st.caption("Thinks Mr. N is a liar/attention seeker.")
        prior_c = st.slider("C: Prior P(S)", 0.0, 1.0, 0.9)
        lc_fa = st.slider("C: P(Claim|Safe)", 0.0, 1.0, 0.99)
        lc_td = st.slider("C: P(Claim|Unsafe)", 0.0, 1.0, 0.90)

    # --- Calculation Phase ---

    agents_data = [
        AgentConfig("Person A", "#ef4444", prior_a, la_fa, la_td, "Trusts N"),
        AgentConfig("Person B", "#22c55e", prior_b, lb_fa, lb_td, "Erratic N"),
        AgentConfig("Person C", "#3b82f6", prior_c, lc_fa, lc_td, "Distrusts N"),
    ]
    agents = [BayesianAgent(conf) for conf in agents_data]

    # --- Display Phase ---

    # Create two columns for the charts
    left_chart, right_chart = st.columns(2)

    with left_chart:
        st.subheader("1. Belief Update Landscape")
        st.caption("How each agent transforms *any* prior into a posterior.")
        fig_landscape = create_belief_landscape(agents)
        st.plotly_chart(fig_landscape, use_container_width=True)

        st.info("""
        **How to read this:**
        * **The Lines:** The "personality" of the agent. Curves below the diagonal mean the agent is skeptical of the drug given the news.
        * **The Dots:** The actual current belief. 
        * **Opinion Switching:** Notice if Person A's dot drops below Person B's dot, even if A started higher!
        """)

    with right_chart:
        st.subheader("2. Pre/Post Snapshot")
        st.caption("Comparison of beliefs before and after the news.")
        fig_bar = create_bar_chart(agents)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Stats Table
        st.markdown("### Numerical Detail")

        # Simple CSS table
        cols = st.columns(3)
        for idx, agent in enumerate(agents):
            with cols[idx]:
                st.markdown(f"**{agent.name}**")
                st.write(f"Prior: `{agent.prior_safe:.2f}`")
                st.write(f"Post: `{agent.posterior_safe:.2f}`")

                # Check for "Opinion Switch" Logic relative to others
                # (Simple check: Did they start high and end low?)
                change = agent.posterior_safe - agent.prior_safe
                color = "green" if change >= 0 else "red"
                st.markdown(f"Change: :{color}[{change:+.2f}]")


if __name__ == "__main__":
    main()
