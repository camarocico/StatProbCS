import streamlit as st
import pandas as pd


def initialize_state():
    """Initializes session state variables if they don't exist."""
    if "red_count" not in st.session_state:
        st.session_state.red_count = 0
    if "black_count" not in st.session_state:
        st.session_state.black_count = 0
    if "history" not in st.session_state:
        st.session_state.history = []


def record_draw(color):
    """Updates counts and history based on the selected color."""
    if color == "Red":
        st.session_state.red_count += 1
    elif color == "Black":
        st.session_state.black_count += 1

    st.session_state.history.append(color)


def reset_state():
    """Resets all counters and history to zero/empty."""
    st.session_state.red_count = 0
    st.session_state.black_count = 0
    st.session_state.history = []


def display_header():
    """Displays the app title and description."""
    st.title("🃏 Card Color Tracker")
    st.markdown("""
        **Instruction:** Draw a card, record its color below, and place it back in the deck.
        The deck contains **50 cards** (Red/Black) in an unknown proportion.
    """)


def display_controls():
    """Displays the input buttons for recording draws."""
    st.subheader("Record a Draw")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🟥 Red", use_container_width=True):
            record_draw("Red")

    with col2:
        if st.button("⬛ Black", use_container_width=True):
            record_draw("Black")

    with col3:
        if st.button("🔄 Reset", type="primary", use_container_width=True):
            reset_state()
            st.rerun()


def display_stats():
    """Calculates and displays the running totals."""
    st.divider()
    st.subheader("Running Totals")

    total_draws = st.session_state.red_count + st.session_state.black_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Red Cards", st.session_state.red_count)
    m2.metric("Black Cards", st.session_state.black_count)
    m3.metric("Total Draws", total_draws)

    return total_draws


def display_visualization(total_draws):
    """Displays the bar chart and history log if data exists."""
    if total_draws > 0:
        st.write("### Proportion Visualization")

        # FIX: Create a DataFrame where 'Red' and 'Black' are separate columns.
        # This allows st.bar_chart to map the 2 colors to the 2 columns correctly.
        data = pd.DataFrame(
            {
                "Red": [st.session_state.red_count / (st.session_state.red_count + st.session_state.black_count)],
                "Black": [st.session_state.black_count / (st.session_state.red_count + st.session_state.black_count)],
            }
        )

        st.bar_chart(data, color=["#FF4B4B", "#1E1E1E"])

        with st.expander("View Draw History"):
            st.write(st.session_state.history[::-1])


def main():
    """Main execution function."""
    # 1. Setup
    st.set_page_config(page_title="Card Tracker", page_icon="🃏")
    initialize_state()

    # 2. UI Layout
    display_header()
    display_controls()

    # 3. Data Display
    total_draws = display_stats()
    display_visualization(total_draws)


if __name__ == "__main__":
    main()
