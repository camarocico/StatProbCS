import math
from math import comb

import numpy as np
import plotly.graph_objects as go
import streamlit as st

P_COIN = 0.5

def simulate_coin_flips(n: int, num_coins: int, p: float = P_COIN) -> np.ndarray:
    """
    Simulate n experiments, each tossing `num_coins` coins with success prob p.
    Returns an array of length n with the proportion of heads.
    """
    # rbinom(n, num_coins, p) / num_coins
    counts = np.random.binomial(num_coins, p, size=n)
    return counts / num_coins

def simulate_dice_rolls(n: int, num_dice: int) -> np.ndarray:
    """
    Simulate n experiments, each rolling `num_dice` fair dice.
    Returns an array of length n with the average of the dice.
    """
    # In R: replicate(n, mean(sample(1:6, num_dice, replace = TRUE)))
    rolls = np.random.randint(1, 7, size=(n, num_dice))
    return rolls.mean(axis=1)

def dice_average_distribution(num_dice: int):
  """
  Approximate distribution of the average of `num_dice` fair dice,
  mimicking the R `dice_average_distribution()`:
  
  - Start with single-die distribution: P(1..6) = 1/6
  - Convolve it with itself `num_dice - 1` times (distribution of the sum)
  - Map sum k in [n, 6n] to average k/n
  """
  single_die = np.full(6, 1 / 6)
  
  # n = 1 trivial case
  if num_dice == 1:
    averages = np.arange(1, 7, dtype=float)
    probabilities = single_die
    return averages, probabilities
  
  dist = single_die.copy()
  for _ in range(2, num_dice + 1):
    dist = np.convolve(dist, single_die)  # distribution of sum of dice
    
  # Sums run from num_dice to 6 * num_dice (inclusive), step 1
  sums = np.arange(num_dice, 6 * num_dice + 1)
  averages = sums / num_dice
  probabilities = dist

  return averages, probabilities

def binomial_probs(num_coins: int, p: float = P_COIN) -> np.ndarray:
  """
  Binomial pmf for X ~ Bin(num_coins, p) over k = 0..num_coins.
  """
  k = np.arange(0, num_coins + 1)
  pmf = np.array(
      [comb(num_coins, ki) * (p**ki) * ((1 - p) ** (num_coins - ki)) for ki in k]
  )
  return pmf

def plot_coin_hist(flip_data: np.ndarray, num_coins: int):
  """
  Plot:
  - histogram of observed proportions of heads
  - red line: Binomial(num_coins, 0.5) density mapped to {0..1}
  """
  if flip_data.size == 0:
    st.info("No coin flips yet. Use the buttons on the left to simulate.")
    return

  # Theoretical Binomial probabilities
  pmf = binomial_probs(num_coins, P_COIN)
  k_vals = np.arange(0, num_coins + 1)
  x_line = k_vals / num_coins
  y_line = pmf

  # Histogram of proportions, normalized to probability
  # -------- Centered histogram -------- #
  bin_size = 1.0 / num_coins
  bins = np.arange(-bin_size / 2, 1.0 + bin_size, bin_size)
  hist_counts, edges = np.histogram(flip_data, bins=bins, density=True)
  centers = (edges[:-1] + edges[1:]) / 2

  hist = go.Bar(
    x=centers,
    y=hist_counts / sum(hist_counts),
    width=bin_size * 0.9,
    marker=dict(color="darkcyan", line=dict(color="black", width=1)),
    name="Observed proportions",
  )

  line = go.Scatter(
    x=x_line,
    y=y_line,
    mode="lines",
    line=dict(color="red", width=2),
    name="Binomial density",
  )

  # x-limits around 0.5 ± 5 * 0.5 / sqrt(num_coins)
  sigma = 0.5 / math.sqrt(num_coins)
  x_min = max(-bin_size / 2, 0.5 - 5 * sigma)
  x_max = min(1.0 + bin_size / 2, 0.5 + 5 * sigma)
  y_max = 1.5 * pmf.max()

  fig = go.Figure(data=[hist, line])
  fig.update_layout(
    title="Coin Flips & Binomial Distribution Density",
    xaxis_title="Proportion of Heads",
    yaxis_title="Density",
    bargap=0.05,
  )
  fig.update_xaxes(range=[x_min, x_max])
  fig.update_yaxes(range=[0, y_max])

  st.plotly_chart(fig, width='stretch')

def plot_dice_hist(roll_data: np.ndarray, num_dice: int):
    """
    Plot:
    - histogram of observed averages of dice
    - red line: theoretical distribution of the average of `num_dice` dice
    """
    if roll_data.size == 0:
        st.info("No dice rolls yet. Use the buttons on the left to simulate.")
        return

    avg_vals, probs = dice_average_distribution(num_dice)

    # -------- Centered histogram -------- #
    bin_size = 1.0 / num_dice
    bins = np.arange(-bin_size / 2, 6.0 + bin_size, bin_size)
    hist_counts, edges = np.histogram(roll_data, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    hist_counts, edges = np.histogram(roll_data, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    hist = go.Bar(
        x=centers,
        y=hist_counts / sum(hist_counts),
        width=bin_size * 0.9,
        marker=dict(color="darkcyan", line=dict(color="black", width=1)),
        name="Observed averages",
    )

    line = go.Scatter(
        x=avg_vals,
        y=probs,
        mode="lines",
        line=dict(color="red", width=2),
        name="Theoretical density",
    )

    # x-limits around 0.5 ± 5 * 0.5 / sqrt(num_coins)
    sigma = 3.5 / math.sqrt(num_dice)
    x_min = max(1.0 - bin_size / 2, 3.5 - 5 * sigma)
    x_max = min(6.0 + bin_size / 2, 3.5 + 5 * sigma)
    y_max = 1.5 * probs.max()

    fig = go.Figure(data=[hist, line])
    fig.update_layout(
        title="Dice Rolls & Distribution Density",
        xaxis_title="Average of Top Faces",
        yaxis_title="Density",
        bargap=0.05,
    )
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[0, y_max])

    st.plotly_chart(fig, width='stretch')

def init_state():
    """
    Initialize all session_state variables used in the app.
    """
    # Coin-related state
    if "flip_data" not in st.session_state:
        st.session_state.flip_data = np.array([], dtype=float)
    if "flip_counter" not in st.session_state:
        st.session_state.flip_counter = 0

    # Dice-related state
    if "roll_data" not in st.session_state:
        st.session_state.roll_data = np.array([], dtype=float)
    if "roll_counter" not in st.session_state:
        st.session_state.roll_counter = 0

def main():
    st.set_page_config(page_title="Coin Flipper and Dice Roller", layout="wide")
    init_state()

    st.title("Coin Flipper and Dice Roller")

    tab_coin, tab_dice = st.tabs(["Coin Flipping", "Dice Rolling"])

    # ---------------------- Coin tab ---------------------- #
    with tab_coin:
        col_left, col_right = st.columns([1, 3])

        with col_left:
            st.subheader("Coin Flipping")

            num_coins = st.number_input(
                "Number of coins to toss:",
                min_value=1,
                max_value=1000,
                value=10,
                step=1,
            )

            st.write("")

            flip_once = st.button("Flip Once")
            flip_10 = st.button("Flip 10 times")
            flip_100 = st.button("Flip 100 Times")
            flip_1m = st.button("Flip 1 Million Times")
            reset_coins = st.button("Reset")

            # Handle actions
            if flip_once:
                new_data = simulate_coin_flips(1, int(num_coins), P_COIN)
                st.session_state.flip_data = np.concatenate(
                    [st.session_state.flip_data, new_data]
                )
                st.session_state.flip_counter += 1

            if flip_10:
                new_data = simulate_coin_flips(10, int(num_coins), P_COIN)
                st.session_state.flip_data = np.concatenate(
                    [st.session_state.flip_data, new_data]
                )
                st.session_state.flip_counter += 10

            if flip_100:
                new_data = simulate_coin_flips(100, int(num_coins), P_COIN)
                st.session_state.flip_data = np.concatenate(
                    [st.session_state.flip_data, new_data]
                )
                st.session_state.flip_counter += 100

            if flip_1m:
                # Warning: this can be heavy for large num_coins, but matches Shiny behaviour
                new_data = simulate_coin_flips(1_000_000, int(num_coins), P_COIN)
                st.session_state.flip_data = np.concatenate(
                    [st.session_state.flip_data, new_data]
                )
                st.session_state.flip_counter += 1_000_000

            if reset_coins:
                st.session_state.flip_data = np.array([], dtype=float)
                st.session_state.flip_counter = 0

            st.markdown("**Total Flips:**")
            st.write(f"{st.session_state.flip_counter}")

        with col_right:
            plot_coin_hist(st.session_state.flip_data, int(num_coins))

    # ---------------------- Dice tab ---------------------- #
    with tab_dice:
        col_left, col_right = st.columns([1, 3])

        with col_left:
            st.subheader("Dice Rolling")

            num_dice = st.number_input(
                "Number of dice to roll:",
                min_value=1,
                max_value=1000,
                value=10,
                step=1,
            )

            st.write("")

            roll_once = st.button("Roll Once", key="roll1")
            roll_10 = st.button("Roll 10 times", key="roll10")
            roll_100 = st.button("Roll 100 Times", key="roll100")
            roll_1m = st.button("Roll 1 Million Times", key="roll1m")
            reset_dice = st.button("Reset", key="reset_dice")

            if roll_once:
                new_data = simulate_dice_rolls(1, int(num_dice))
                st.session_state.roll_data = np.concatenate(
                    [st.session_state.roll_data, new_data]
                )
                st.session_state.roll_counter += 1

            if roll_10:
                new_data = simulate_dice_rolls(10, int(num_dice))
                st.session_state.roll_data = np.concatenate(
                    [st.session_state.roll_data, new_data]
                )
                st.session_state.roll_counter += 10

            if roll_100:
                new_data = simulate_dice_rolls(100, int(num_dice))
                st.session_state.roll_data = np.concatenate(
                    [st.session_state.roll_data, new_data]
                )
                st.session_state.roll_counter += 100

            if roll_1m:
                new_data = simulate_dice_rolls(1_000_000, int(num_dice))
                st.session_state.roll_data = np.concatenate(
                    [st.session_state.roll_data, new_data]
                )
                st.session_state.roll_counter += 1_000_000

            if reset_dice:
                st.session_state.roll_data = np.array([], dtype=float)
                st.session_state.roll_counter = 0

            st.markdown("**Total Rolls:**")
            st.write(f"{st.session_state.roll_counter}")

        with col_right:
            plot_dice_hist(st.session_state.roll_data, int(num_dice))

if __name__ == "__main__":
    main()
