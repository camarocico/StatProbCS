import streamlit as st
import numpy as np
import plotly.graph_objs as go
from typing import List, Tuple

# --------- Core computations (Log-Domain / dB) ---------


def to_db(p: float) -> float:
    """Convert probability to decibels safely."""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    return 10.0 * np.log10(p / (1 - p))


def from_db(db: float) -> float:
    """Convert decibels to raw value (unnormalized)."""
    return 10.0 ** (db / 10.0)


def normalize_priors_to_scores(
    ev_b: float, ev_g: float, ev_r: float, include_rogue: bool
) -> List[float]:
    """
    Takes user input Evidence (dB) and converts them to normalized log-probability scores (dB).
    Score S_k = 10 * log10( P(H_k) )
    """

    # 1. Convert Evidence (dB) -> Odds -> Unnormalized Prob
    # O = P / (1-P) = 10^(E/10)  =>  P = O / (1+O)
    def get_prob_from_evidence(e):
        odds = from_db(e)
        return odds / (1.0 + odds)

    p_b = get_prob_from_evidence(ev_b)
    p_g = get_prob_from_evidence(ev_g)
    p_r = get_prob_from_evidence(ev_r) if include_rogue else 0.0

    # 2. Normalize
    total = p_b + p_g + p_r
    if total == 0:
        total = 1.0  # Safety

    p_b /= total
    p_g /= total
    p_r /= total

    # 3. Return as Scores (dB) i.e., 10*log10(P)
    # We use a small epsilon for log(0) safety, though practically p_r=0 is handled by mask
    scores = [to_db(p_b + 1e-20), to_db(p_g + 1e-20), to_db(p_r + 1e-20)]
    return scores


def update_scores(
    current_scores: List[float],
    result_type: str,
    rates: List[float],
    include_rogue: bool,
) -> List[float]:
    """
    Updates the scores recursively.
    New Score = Old Score + 10*log10(Likelihood)
    """
    new_scores = []

    # Identify which likelihood to use based on result 'G' (Good) or 'B' (Bad)
    # rates order: [Bad, Good, Rogue]

    for i, score in enumerate(current_scores):
        if i == 2 and not include_rogue:
            new_scores.append(score)  # Keep rogue score as is (usually -inf)
            continue

        # P(D|H)
        rate = rates[i]
        likelihood = rate if result_type == "B" else (1.0 - rate)

        # Add Evidence (dB)
        weight_db = to_db(likelihood)
        new_scores.append(score + weight_db)

    return new_scores


def compute_display_metrics(
    scores: List[float], include_rogue: bool
) -> Tuple[List[float], List[float]]:
    """
    Converts internal Scores (dB) -> Normalized Probabilities -> Evidence (dB).
    Returns (Evidences, Probabilities)
    """
    # 1. Log-Sum-Exp trick in base 10 to normalize
    # P_k = 10^(S_k/10) / Sum(10^(S_j/10))

    raw_vals = [from_db(s) for s in scores]

    # If rogue is excluded, zero it out for sum calculation to be clean
    if not include_rogue:
        raw_vals[2] = 0.0

    total_val = sum(raw_vals)
    if total_val == 0:
        total_val = 1.0

    probs = [v / total_val for v in raw_vals]

    # 2. Convert Prob -> Evidence
    # E = 10 * log10( P / (1-P) )
    evidences = []
    for p in probs:
        # if p >= 1.0 - 1e-9:
        #     evidences.append(100.0)  # Cap at 100 dB
        # elif p <= 1e-9:
        #     evidences.append(-100.0)  # Floor at -100 dB
        # else:
        print(p / (1.0 - p))
        evidences.append(10.0 * np.log10(p / (1.0 - p)))

    return evidences, probs


# --------- State management ---------


def init_state():
    # Store the full history of scores (dB) to allow efficient plotting
    # Format: List of [Score_B, Score_G, Score_R]
    if "score_history" not in st.session_state:
        st.session_state.score_history = []
    if "priors_initialized" not in st.session_state:
        st.session_state.priors_initialized = False


# --------- Streamlit app ---------


def main():
    st.set_page_config(
        page_title="Bayesian Widget Tester (Exact)",
        layout="wide",
    )
    init_state()

    st.title("Bayesian Hypothesis Testing: Recursive Evidence")

    # --- Sidebar: Configuration ---
    st.sidebar.header("Configuration")

    include_rogue = st.sidebar.checkbox("Include Rogue Machine ($H_r$)?", value=False)
    st.sidebar.divider()

    st.sidebar.subheader("1. Defect Rates")
    # Using small epsilon to strictly avoid log(0) if user sets rate to 0.0 or 1.0
    rate_b = st.sidebar.number_input(
        "Bad Machine Rate ($H_b$)",
        min_value=0.01,
        max_value=0.99,
        value=0.66,
        step=0.01,
    )
    rate_g = st.sidebar.number_input(
        "Good Machine Rate ($H_g$)",
        min_value=0.01,
        max_value=0.99,
        value=0.16,
        step=0.01,
    )

    rate_r = 0.95
    if include_rogue:
        rate_r = st.sidebar.number_input(
            "Rogue Machine Rate ($H_r$)",
            min_value=0.01,
            max_value=0.99,
            value=0.95,
            step=0.01,
        )

    st.sidebar.divider()
    st.sidebar.subheader("2. Prior Evidence (dB)")

    ev_b_prior = st.sidebar.number_input(
        "Prior Evidence Hb (dB)", value=-10.0, step=1.0
    )
    ev_g_prior = st.sidebar.number_input("Prior Evidence Hg (dB)", value=10.0, step=1.0)

    ev_r_prior = -60.0
    if include_rogue:
        ev_r_prior = st.sidebar.number_input(
            "Prior Evidence Hr (dB)", value=-60.0, step=5.0
        )

    # --- Initialize / Re-Initialize Scores based on Inputs ---
    # We maintain a base score derived from priors
    base_scores = normalize_priors_to_scores(
        ev_b_prior, ev_g_prior, ev_r_prior, include_rogue
    )

    # We track clicks separately to allow "Replay" with new parameters
    if "click_history" not in st.session_state:
        st.session_state.click_history = []

    # --- Main Layout ---
    col_left, col_right = st.columns([2, 5])

    with col_left:
        st.subheader("Inspection Station")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Add Good (+1)", type="primary"):
                st.session_state.click_history.append("G")
        with c2:
            if st.button("Add Bad (+1)", type="primary"):
                st.session_state.click_history.append("B")
        with c3:
            if st.button("Reset", type="secondary"):
                st.session_state.click_history = []

        # --- Recompute Full History (Exact Recursive Update) ---
        # We recompute on every render to ensure consistency with Sidebar Sliders

        rates = [rate_b, rate_g, rate_r]

        # Start with priors
        history_scores = [base_scores]

        # Apply updates sequentially
        curr_s = base_scores
        for click in st.session_state.click_history:
            curr_s = update_scores(curr_s, click, rates, include_rogue)
            history_scores.append(curr_s)

        # Get Current State Metrics
        curr_ev, curr_probs = compute_display_metrics(curr_s, include_rogue)

        curr_good = st.session_state.click_history.count("G")
        curr_bad = st.session_state.click_history.count("B")

        st.divider()
        st.metric("Good Widgets", curr_good)
        st.metric("Bad Widgets", curr_bad)

        st.divider()
        st.subheader("Current Belief")

        st.markdown(
            f"**$H_g$ (Good)**: {curr_probs[1]:.2%}  _(Evidence: {curr_ev[1]:.2f} dB)_"
        )
        st.progress(curr_probs[1])

        st.markdown(
            f"**$H_b$ (Bad)**: {curr_probs[0]:.2%} _(Evidence: {curr_ev[0]:.2f} dB)_"
        )
        st.progress(curr_probs[0])

        if include_rogue:
            st.markdown(
                f"**$H_r$ (Rogue)**: {curr_probs[2]:.2%} _(Evidence: {curr_ev[2]:.2f} dB)_"
            )
            st.progress(curr_probs[2])

    with col_right:
        st.subheader("Evidence Trajectory (Exact dB)")

        # Extract Evidence History for plotting
        # We need to convert the score history -> evidence history
        hist_ev_b, hist_ev_g, hist_ev_r = [], [], []

        for s_step in history_scores:
            evs, _ = compute_display_metrics(s_step, include_rogue)
            hist_ev_b.append(evs[0])
            hist_ev_g.append(evs[1])
            hist_ev_r.append(evs[2])

        x_vals = list(range(len(hist_ev_b)))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=hist_ev_g,
                mode="lines+markers",
                name="Hg (Good)",
                line=dict(color="green"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=hist_ev_b,
                mode="lines+markers",
                name="Hb (Bad)",
                line=dict(color="orange"),
            )
        )

        if include_rogue:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=hist_ev_r,
                    mode="lines+markers",
                    name="Hr (Rogue)",
                    line=dict(color="red", dash="dot"),
                )
            )

        fig.update_layout(
            title="Evidence Evolution",
            xaxis_title="Number of Widgets Tested",
            yaxis_title="Evidence $10 \log_{10} \frac{P}{1-P}$ (dB)",
            template="plotly_white",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
