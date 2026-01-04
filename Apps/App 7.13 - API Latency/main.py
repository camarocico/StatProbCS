import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- App Configuration ---
st.set_page_config(page_title="DevOps Hypothesis Test", layout="centered")

st.title("🖥️ API Latency Hypothesis Tester")
st.markdown("""
**Scenario:** You are testing if a new code deployment has altered the average response time of your API.
* **Null Hypothesis ($H_0$):** Latency = 50 ms (SLA compliant).
* **Alternative ($H_1$):** Latency $\\neq$ 50 ms.

Adjust the **observed sample mean** and **variance** below to see if you should trigger an alert!
""")

st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("1. SLA Parameters ($H_0$)")
mu_0 = st.sidebar.number_input("Target Latency (ms)", value=50.0, step=0.1)
n = st.sidebar.number_input("Sample Size (Requests)", value=64, step=1)

st.sidebar.header("2. Test Data (Modify These)")
# User inputs Variance, we calculate SD
initial_sd = 8.0
initial_var = initial_sd**2
sigma_sq = st.sidebar.slider(
    "Population Variance ($\sigma^2$)",
    min_value=1.0,
    max_value=150.0,
    value=float(initial_var),
    step=0.5,
)
sigma = np.sqrt(sigma_sq)
st.sidebar.caption(f"Standard Deviation ($\sigma$): **{sigma:.2f} ms**")

sample_mean = st.sidebar.slider(
    "Observed Sample Mean (ms)", min_value=45.0, max_value=55.0, value=53.0, step=0.1
)

st.sidebar.header("3. Risk Tolerance")
alpha = st.sidebar.selectbox(
    "Significance Level ($\\alpha$)", options=[0.01, 0.05, 0.10], index=1
)

# --- Calculations ---
se = sigma / np.sqrt(n)
z_score = (sample_mean - mu_0) / se

# Critical values
z_crit = stats.norm.ppf(1 - alpha / 2)
x_crit_right = mu_0 + (z_crit * se)
x_crit_left = mu_0 - (z_crit * se)

p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

# X-axis range
x_axis = np.linspace(mu_0 - 4 * se, mu_0 + 4 * se, 1000)
y_axis = stats.norm.pdf(x_axis, mu_0, se)

# Plot H0 Distribution
ax.plot(x_axis, y_axis, color="#2c3e50", lw=2, label="Expected Latency ($H_0$)")

# Shade Rejection Regions
# Right Tail
x_right = np.linspace(x_crit_right, mu_0 + 4 * se, 200)
ax.fill_between(
    x_right,
    stats.norm.pdf(x_right, mu_0, se),
    color="#e74c3c",
    alpha=0.6,
    label="Rejection Region (Anomaly)",
)
# Left Tail
x_left = np.linspace(mu_0 - 4 * se, x_crit_left, 200)
ax.fill_between(x_left, stats.norm.pdf(x_left, mu_0, se), color="#e74c3c", alpha=0.6)

# Plot Observed Mean
line_color = "#e74c3c" if abs(z_score) > z_crit else "#27ae60"
ax.axvline(
    sample_mean,
    color=line_color,
    linestyle="--",
    linewidth=3,
    label=f"Observed Mean ({sample_mean} ms)",
)

ax.set_title(f"Sampling Distribution (n={n})", fontsize=14)
ax.set_xlabel("Latency (ms)")
ax.set_yticks([])  # Hide Y axis numbers for cleanliness
ax.legend(loc="upper right")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)

st.pyplot(fig)

# --- Results ---
st.subheader("Test Results")
c1, c2, c3 = st.columns(3)
c1.metric("Z-Score", f"{z_score:.2f}")
c2.metric("Critical Z", f"±{z_crit:.2f}")

is_reject = abs(z_score) > z_crit
result_text = "REJECT H₀ (Alert!)" if is_reject else "FAIL TO REJECT (Normal)"
result_color = "inverse" if is_reject else "normal"
c3.metric("Decision", result_text, f"p = {p_value:.4f}", delta_color=result_color)

if is_reject:
    st.error(
        f"**Conclusion:** The observed latency ({sample_mean} ms) is significantly different from the SLA ({mu_0} ms). Investigation required."
    )
else:
    st.success(
        f"**Conclusion:** The observed latency ({sample_mean} ms) is within acceptable random variance of the SLA ({mu_0} ms). No action needed."
    )
