import math
from math import comb

import numpy as np
import plotly.graph_objects as go
import streamlit as st

P_DEFECTIVE = 0.5
LAMBDA_PARAMETER = 10.0


def simulate_response_times(
    n: int = 1, sample_size: int = 1, lam: float = LAMBDA_PARAMETER
) -> np.ndarray:
    """
    Simulate n samples, each consisting of `sample_size` exponential(λ) responses.
    Return the distribution of sample means.
    """
    # rexp(sample_size, lambda_parameter) then mean, repeated n times
    # shape (n, sample_size)
    samples = np.random.exponential(scale=1.0 / lam, size=(n, sample_size))
    return samples.mean(axis=1)


def simulate_defective_motherboards(
    n: int = 1, sample_size: int = 1, p: float = P_DEFECTIVE
) -> np.ndarray:
    """
    Simulate n samples, each consisting of `sample_size` motherboards.
    Each motherboard is defective with probability p.

    Return the distribution of sample proportions of defectives.
    """
    # Each row: Binomial(sample_size, p) successes (defectives)
    counts = np.random.binomial(sample_size, p, size=n)
    return counts / sample_size


def plot_rt_hist(
    rt_sample_data: np.ndarray,
    sample_size: int = 1,
    show_clt: bool = True,
    lam: float = LAMBDA_PARAMETER,
):
    """
    Plot the sampling distribution of mean response times:
    - Centered histogram of observed sample means (density)
    - Optional CLT Normal curve with mean = 1/λ, sd = 1/(λ√n)
    """
    if rt_sample_data.size == 0:
        st.info(
            "No samples yet. Use the buttons on the left to simulate response times."
        )
        return

    mu = 1.0 / lam
    sigma = 1.0 / lam / math.sqrt(sample_size)

    # Histogram breaks: seq(mu - 10σ, mu + 10σ, 0.2σ)
    bin_width = 0.2 * sigma
    bins = np.arange(mu - 10.0 * sigma, mu + 10.0 * sigma + bin_width, bin_width)
    hist_counts, edges = np.histogram(rt_sample_data, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Histogram as centered bars
    hist_bar = go.Bar(
        x=centers,
        y=hist_counts,
        width=bin_width * 0.9,
        marker=dict(color="darkcyan", line=dict(color="black", width=1)),
        name="Observed means",
    )

    traces = [hist_bar]

    if show_clt:
        # CLT Normal curve over mu ± 5σ (like in the R code)
        x = np.arange(mu - 5.0 * sigma, mu + 5.0 * sigma, 0.1 * sigma)
        rt_clt = (
            1.0
            / (sigma * math.sqrt(2.0 * math.pi))
            * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        )

        line = go.Scatter(
            x=x,
            y=rt_clt,
            mode="lines",
            line=dict(color="red", width=2),
            name="CLT Normal approx",
        )
        traces.append(line)
        y_max = 1.1 * max(hist_counts.max(), rt_clt.max())
    else:
        y_max = 1.1 * hist_counts.max()

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Mean Response Time Sampling Distribution & CLT Approximation",
        xaxis_title="Mean Response Times",
        yaxis_title="Density",
        bargap=0.05,
    )
    fig.update_yaxes(range=[0, y_max])

    st.plotly_chart(fig, width="stretch")


def plot_dm_hist(
    dm_sample_data: np.ndarray, sample_size: int = 1, p: float = P_DEFECTIVE
):
    """
    Plot the sampling distribution of proportion defective:
    - Centered histogram of observed sample proportions (density)
    - Red line: binomial pmf mapped to density over k/n
    """
    if dm_sample_data.size == 0:
        st.info(
            "No samples yet. Use the buttons on the left to simulate defective motherboards."
        )
        return

    # Histogram bins centered at k/n for k = 0..n
    bin_size = 1.0 / sample_size
    # Edges from -0.5/n to 1+0.5/n to center bins on 0, 1/n, ..., 1
    bins = np.arange(-0.5 * bin_size, 1.0 + 1.5 * bin_size, bin_size)
    hist_counts, edges = np.histogram(dm_sample_data, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    hist_bar = go.Bar(
        x=centers,
        y=hist_counts,
        width=bin_size * 0.9,
        marker=dict(color="darkcyan", line=dict(color="black", width=1)),
        name="Observed proportions",
    )

    # Binomial pmf: k ~ Bin(sample_size, p), x = k/n
    # Normal distribution from the Central Limit Theorem:
    mu = p
    sigma = math.sqrt(p * (1 - p) / sample_size)
    normal_x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    normal_y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((normal_x - mu) / sigma) ** 2
    )

    k_vals = np.arange(0, sample_size + 1)
    pmf = np.array(
        [comb(sample_size, k) * p**k * (1 - p) ** (sample_size - k) for k in k_vals]
    )
    # Convert pmf to density for step size 1/n
    density = pmf * sample_size
    x_line = k_vals / sample_size

    line = go.Scatter(
        x=x_line,
        y=density,
        mode="lines",
        line=dict(color="red", width=2),
        name="Binomial density",
    )

    lineclt = go.Scatter(
        x=normal_x,
        y=normal_y,
        mode="lines",
        line=dict(color="green", width=2),
        name="CLT Normal approx",
    )

    y_max = 1.1 * max(hist_counts.max(), density.max(), normal_y.max())

    fig = go.Figure(data=[hist_bar, line, lineclt])
    fig.update_layout(
        title="Defective Motherboards: Sampling Distribution, Binomial Model and CLT Approximation",
        xaxis_title="Proportion Defective",
        yaxis_title="Density",
        bargap=0.05,
    )
    fig.update_yaxes(range=[0, y_max])

    st.plotly_chart(fig, width="stretch")


def init_state():
    # Response time sampling
    if "rt_data" not in st.session_state:
        st.session_state.rt_data = np.array([], dtype=float)
    if "rt_counter" not in st.session_state:
        st.session_state.rt_counter = 0
    if "rt_show_clt" not in st.session_state:
        st.session_state.rt_show_clt = False  # CLT line toggle

    # Defective motherboard sampling
    if "dm_data" not in st.session_state:
        st.session_state.dm_data = np.array([], dtype=float)
    if "dm_counter" not in st.session_state:
        st.session_state.dm_counter = 0


def main():
    st.set_page_config(page_title="Simulation App", layout="wide")
    init_state()

    st.title("Simulation App")

    tab_rt, tab_dm = st.tabs(
        ["Mean Response Times Simulation", "Defective Motherboard Simulation"]
    )

    # ---------------------- Response times tab ---------------------- #
    with tab_rt:
        col_left, col_right = st.columns([1, 3])

        with col_left:
            st.subheader("Mean Response Times Simulation")

            rt_sample_size = st.number_input(
                "Number of responses to average over:",
                min_value=1,
                max_value=10_000,
                value=10,
                step=1,
            )
            lambda_exp = st.slider(
                "Rate parameter:",
                min_value=0.01,
                max_value=100.0,
                value=10.0,
                step=1.0,
            )

            st.write("")

            rt_sample1 = st.button("Sample Once")
            rt_sample10 = st.button("Sample 10 Times")
            rt_sample100 = st.button("Sample 100 Times")
            rt_sample1m = st.button("Sample 1 Million")
            rt_plot_clt = st.button("Plot CLT / Toggle CLT")
            rt_reset = st.button("Reset")

            # Actions
            if rt_sample1:
                new_data = simulate_response_times(
                    1, int(rt_sample_size), lam=lambda_exp
                )
                st.session_state.rt_data = np.concatenate(
                    [st.session_state.rt_data, new_data]
                )
                st.session_state.rt_counter += 1

            if rt_sample10:
                new_data = simulate_response_times(
                    10, int(rt_sample_size), lam=lambda_exp
                )
                st.session_state.rt_data = np.concatenate(
                    [st.session_state.rt_data, new_data]
                )
                st.session_state.rt_counter += 10

            if rt_sample100:
                new_data = simulate_response_times(
                    100, int(rt_sample_size), lam=lambda_exp
                )
                st.session_state.rt_data = np.concatenate(
                    [st.session_state.rt_data, new_data]
                )
                st.session_state.rt_counter += 100

            if rt_sample1m:
                new_data = simulate_response_times(
                    1_000_000, int(rt_sample_size), lam=lambda_exp
                )
                st.session_state.rt_data = np.concatenate(
                    [st.session_state.rt_data, new_data]
                )
                st.session_state.rt_counter += 1_000_000

            if rt_plot_clt:
                # Toggle CLT line visibility
                st.session_state.rt_show_clt = not st.session_state.rt_show_clt

            if rt_reset:
                st.session_state.rt_data = np.array([], dtype=float)
                st.session_state.rt_counter = 0
                st.session_state.rt_show_clt = False

            st.markdown("**Total #Samples:**")
            st.write(f"{st.session_state.rt_counter}")

        with col_right:
            plot_rt_hist(
                st.session_state.rt_data,
                int(rt_sample_size),
                st.session_state.rt_show_clt,
                lam=lambda_exp,
            )

    # ---------------------- Defective motherboard tab ---------------------- #
    with tab_dm:
        col_left, col_right = st.columns([1, 3])

        with col_left:
            st.subheader("Defective Motherboard Simulation")

            dm_sample_size = st.number_input(
                "Sample Size:",
                min_value=1,
                max_value=1000,
                value=1,
                step=1,
            )
            p = st.slider(
                "Probability of defective motherboard:",
                min_value=0.01,
                max_value=0.99,
                value=0.50,
                step=0.01,
            )

            st.write("")

            dm_sample1 = st.button("Sample Once", key="dm1")
            dm_sample10 = st.button("Sample 10 Times", key="dm10")
            dm_sample100 = st.button("Sample 100 Times", key="dm100")
            dm_sample1m = st.button("Sample 1 Million", key="dm1m")
            dm_reset = st.button("Reset", key="dm_reset")

            if dm_sample1:
                new_data = simulate_defective_motherboards(1, int(dm_sample_size), p)
                st.session_state.dm_data = np.concatenate(
                    [st.session_state.dm_data, new_data]
                )
                st.session_state.dm_counter += 1

            if dm_sample10:
                new_data = simulate_defective_motherboards(10, int(dm_sample_size), p)
                st.session_state.dm_data = np.concatenate(
                    [st.session_state.dm_data, new_data]
                )
                st.session_state.dm_counter += 10

            if dm_sample100:
                new_data = simulate_defective_motherboards(100, int(dm_sample_size), p)
                st.session_state.dm_data = np.concatenate(
                    [st.session_state.dm_data, new_data]
                )
                st.session_state.dm_counter += 100

            if dm_sample1m:
                new_data = simulate_defective_motherboards(
                    1_000_000, int(dm_sample_size), p
                )
                st.session_state.dm_data = np.concatenate(
                    [st.session_state.dm_data, new_data]
                )
                st.session_state.dm_counter += 1_000_000

            if dm_reset:
                st.session_state.dm_data = np.array([], dtype=float)
                st.session_state.dm_counter = 0

            st.markdown("**Total Samples:**")
            st.write(f"{st.session_state.dm_counter}")

        with col_right:
            plot_dm_hist(st.session_state.dm_data, int(dm_sample_size), p)


if __name__ == "__main__":
    main()
