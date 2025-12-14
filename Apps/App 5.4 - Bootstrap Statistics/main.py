import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class BootstrapModel:
    """
    Handles the statistical logic: Data storage, resampling, and CI calculation.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.n = len(data)
        self.bootstrap_means = None
        self.ci_lower = None
        self.ci_upper = None
        self.confidence_level = 0.95

    def get_summary_stats(self) -> dict:
        """Returns basic descriptive statistics of the original sample."""
        return {
            "mean": np.mean(self.data),
            "std": np.std(self.data, ddof=1),
            "min": np.min(self.data),
            "max": np.max(self.data),
        }

    def get_theoretical_ci(self, confidence_level: float = 0.95):
        """
        Calculates the standard T-distribution Confidence Interval
        (Parametric approach) for comparison.
        """
        mean = np.mean(self.data)
        sem = stats.sem(self.data)  # Standard Error of Mean
        # t.interval requires alpha, degrees of freedom, loc (mean), scale (std err)
        return stats.t.interval(confidence_level, df=self.n - 1, loc=mean, scale=sem)

    def run_simulation(self, n_iterations: int = 10000):
        """
        Performs the bootstrap resampling.
        Uses vectorization for speed, falls back to loops for massive memory usage.
        """
        # Memory safety check: limit vectorized approach to ~50M elements
        if n_iterations * self.n < 50_000_000:
            # Vectorized (Fast)
            # Create a matrix of (n_iterations x sample_size)
            samples = np.random.choice(self.data, (n_iterations, self.n), replace=True)
            self.bootstrap_means = np.mean(samples, axis=1)
        else:
            # Memory Safe Loop (Slower but crash-resistant)
            means = []
            progress_bar = st.progress(0)
            step = max(1, n_iterations // 100)

            for i in range(n_iterations):
                sample = np.random.choice(self.data, size=self.n, replace=True)
                means.append(np.mean(sample))
                if i % step == 0:
                    progress_bar.progress((i + 1) / n_iterations)

            progress_bar.empty()
            self.bootstrap_means = np.array(means)

    def calculate_ci(self, confidence_level: float = 0.95) -> tuple:
        """Calculates percentile-based CI from the simulation results."""
        self.confidence_level = confidence_level
        if self.bootstrap_means is None:
            raise ValueError("Simulation not run yet.")

        lower_p = (1 - confidence_level) / 2 * 100
        upper_p = (1 + confidence_level) / 2 * 100

        self.ci_lower = np.percentile(self.bootstrap_means, lower_p)
        self.ci_upper = np.percentile(self.bootstrap_means, upper_p)

        return self.ci_lower, self.ci_upper, np.mean(self.bootstrap_means)


class BootstrapApp:
    """
    Handles the Streamlit UI layout and interaction.
    """

    def __init__(self):
        st.set_page_config(page_title="Bootstrap Master", page_icon="🎓", layout="wide")
        self.data = None
        self.model = None

    def render_sidebar(self):
        st.sidebar.title("Configuration")

        st.sidebar.subheader("1. Data Input")
        data_mode = st.sidebar.radio(
            "Data Source:", ["Manual Entry", "Generate Random Data"]
        )

        if data_mode == "Manual Entry":
            default_val = "85, 90, 78, 88, 92, 82, 76, 91, 20"
            input_str = st.sidebar.text_area(
                "Enter values (comma separated)", value=default_val, height=100
            )
            self.data = self._parse_data(input_str)
        else:
            dist_type = st.sidebar.selectbox(
                "Distribution Type",
                ["Normal (Bell Curve)", "Exponential (Skewed)", "Uniform"],
            )
            n_samples = st.sidebar.slider("Sample Size (n)", 10, 500, 30)

            if st.sidebar.button("Generate New Sample"):
                if dist_type.startswith("Normal"):
                    self.data = np.random.normal(loc=50, scale=10, size=n_samples)
                elif dist_type.startswith("Exponential"):
                    self.data = np.random.exponential(scale=10, size=n_samples)
                else:
                    self.data = np.random.uniform(0, 100, size=n_samples)
                st.session_state["generated_data"] = self.data
            elif "generated_data" in st.session_state:
                self.data = st.session_state["generated_data"]
            else:
                # Default init
                self.data = np.random.normal(50, 10, 30)

        st.sidebar.markdown("---")
        st.sidebar.subheader("2. Simulation Settings")
        self.n_iters = st.sidebar.number_input(
            "Bootstrap Iterations", 1000, 1000000, 10000, step=1000
        )
        self.conf_level = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        self.bins = st.sidebar.slider("Histogram Bins", 10, 200, 40)

    def _parse_data(self, input_str):
        try:
            return np.array(
                [float(x.strip()) for x in input_str.split(",") if x.strip()]
            )
        except ValueError:
            return None

    def render_header(self):
        st.title("🎓 Interactive Bootstrap Explorer")
        st.markdown("""
        This tool demonstrates how **Bootstrapping** uses resampling to estimate the sampling distribution 
        of a statistic (like the mean) without needing complex formulas or assumptions about normality.
        """)

        with st.expander("📘 How it works (The Algorithm)"):
            st.markdown("""
            1. **Take the original sample** of size $n$.
            2. **Resample**: Draw $n$ values from the original sample *with replacement*.
            3. **Calculate**: Compute the mean (or other statistic) of this new resample.
            4. **Repeat**: Do this thousands of times.
            5. **Analyze**: The distribution of these thousands of means approximates the true sampling distribution.
            """)

    def render_analysis(self):
        if self.data is None or len(self.data) < 2:
            st.warning("Please enter valid numeric data (at least 2 points).")
            return

        # Initialize Model
        self.model = BootstrapModel(self.data)
        stats_dict = self.model.get_summary_stats()

        # --- Layout: Original Data Analysis ---
        st.subheader("1. Original Sample Analysis")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Sample Statistics")
            st.metric("Sample Size (n)", int(len(self.data)))
            st.metric("Observed Mean", f"{stats_dict['mean']:.4f}")
            st.metric("Observed Std Dev", f"{stats_dict['std']:.4f}")

        with col2:
            fig_orig, ax_orig = plt.subplots(figsize=(8, 3))
            ax_orig.hist(self.data, bins=15, color="gray", alpha=0.6, edgecolor="black")
            ax_orig.axvline(
                stats_dict["mean"], color="red", linestyle="--", label="Sample Mean"
            )
            ax_orig.set_title("Distribution of Original Sample Data")
            ax_orig.legend()
            st.pyplot(fig_orig)

        st.divider()

        # --- Layout: Bootstrap Simulation ---
        st.subheader("2. Bootstrap Simulation")

        if st.button("🚀 Run Bootstrap Simulation", type="primary"):
            with st.spinner(f"Resampling {self.n_iters:,} times..."):
                self.model.run_simulation(self.n_iters)
                ci_low, ci_high, boot_mean = self.model.calculate_ci(self.conf_level)

                # Get Theoretical CI for comparison
                theo_low, theo_high = self.model.get_theoretical_ci(self.conf_level)

                # --- Results Metrics ---
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Bootstrap Mean",
                    f"{boot_mean:.4f}",
                    delta=f"{boot_mean - stats_dict['mean']:.4f}",
                )
                m2.metric(
                    f"Lower Bound ({self.conf_level * 100:.0f}%)", f"{ci_low:.4f}"
                )
                m3.metric(
                    f"Upper Bound ({self.conf_level * 100:.0f}%)", f"{ci_high:.4f}"
                )

                # --- Pedagogical Comparison Table ---
                st.markdown("#### 🆚 Method Comparison")
                st.markdown(
                    "Notice how the intervals differ if your original data was skewed."
                )

                comp_data = {
                    "Method": [
                        "Bootstrap (Percentile)",
                        "Theoretical (T-distribution)",
                    ],
                    "Lower Bound": [ci_low, theo_low],
                    "Upper Bound": [ci_high, theo_high],
                    "Width": [ci_high - ci_low, theo_high - theo_low],
                }
                st.dataframe(comp_data, use_container_width=True)

                # --- Main Visualization ---
                self.plot_results(boot_mean, ci_low, ci_high, theo_low, theo_high)

    def plot_results(self, boot_mean, ci_low, ci_high, theo_low, theo_high):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram of Bootstrap Means
        ax.hist(
            self.model.bootstrap_means,
            bins=self.bins,
            color="skyblue",
            edgecolor="white",
            density=True,
            alpha=0.8,
            label="Bootstrap Distribution",
        )

        # Bootstrap CI Lines
        ax.axvline(
            ci_low,
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f"Bootstrap CI ({self.conf_level * 100:.0f}%)",
        )
        ax.axvline(ci_high, color="blue", linestyle="-", linewidth=2)

        # Theoretical CI Lines (Dashed)
        ax.axvline(
            theo_low,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Theoretical T-Test CI",
        )
        ax.axvline(theo_high, color="red", linestyle="--", linewidth=2, alpha=0.7)

        # Sample Mean
        ax.axvline(
            boot_mean,
            color="black",
            linestyle=":",
            linewidth=2,
            label="Mean of Resamples",
        )

        ax.set_title(
            f"Sampling Distribution of the Mean ({self.n_iters:,} iterations)",
            fontsize=14,
        )
        ax.set_xlabel("Mean Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)

        st.info("""
        **What are we looking at?** The blue histogram represents the *uncertainty* of the mean. 
        If the data is skewed, the Bootstrap CI (Blue lines) might be shifted compared to the Theoretical CI (Red dashed lines), 
        showing that Bootstrapping can better handle non-normal data.
        """)

    def run(self):
        self.render_sidebar()
        self.render_header()
        self.render_analysis()


if __name__ == "__main__":
    app = BootstrapApp()
    app.run()
