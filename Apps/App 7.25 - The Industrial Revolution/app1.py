import math

import numpy as np
import plotly.graph_objs as go
import streamlit as st


def eh(n_good: int, n_bad: int):
    """
    Evidence (in dB) for Hb and Hg given counts of good/bad widgets,
    matching the original Dash implementation.
    """
    EHb = -10 + 20.0 * np.log10(2) * n_bad + 10.0 * n_good * np.log10(2.0 / 5.0)
    EHg = 10 - 20.0 * np.log10(2) * n_bad + 10.0 * n_good * np.log10(5.0 / 2.0)
    return EHb, EHg


def init_state():
    """
    Initialize Streamlit session state to reproduce the Dash behaviour:
    - start with n_good = 0, n_bad = 0
    - and an initial evidence point at (EHb, EHg) for (0, 0).
    """
    if "n_good" not in st.session_state:
        st.session_state.n_good = 0
    if "n_bad" not in st.session_state:
        st.session_state.n_bad = 0
    if "EHb_list" not in st.session_state or "EHg_list" not in st.session_state:
        EHb_init, EHg_init = eh(0, 0)  # corresponds to Dash's first callback at (0,0)
        st.session_state.EHb_list = [EHb_init]
        st.session_state.EHg_list = [EHg_init]


def main():
    st.set_page_config(
        page_title="Evidences for Hb and Hg",
        layout="wide",
    )

    init_state()

    st.title("Evidences for H$_b$ and H$_g$")

    # Two columns: left for controls and stats, right for the graph
    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader("Widget inspection")

        # Buttons (equivalent to Dash's Good/Bad Widget buttons)
        clicked_good = st.button("Good Widget")
        clicked_bad = st.button("Bad Widget")

        # Update counts and evidence history when buttons are clicked
        if clicked_good:
            st.session_state.n_good += 1
            EHbD, EHgD = eh(st.session_state.n_good, st.session_state.n_bad)
            st.session_state.EHb_list.append(EHbD)
            st.session_state.EHg_list.append(EHgD)

        if clicked_bad:
            st.session_state.n_bad += 1
            EHbD, EHgD = eh(st.session_state.n_good, st.session_state.n_bad)
            st.session_state.EHb_list.append(EHbD)
            st.session_state.EHg_list.append(EHgD)

        # Current counts
        n_good = st.session_state.n_good
        n_bad = st.session_state.n_bad
        total = n_good + n_bad

        st.write(f"Good widgets: **{n_good}**")
        st.write(f"Bad widgets: **{n_bad}**")
        st.write(f"Total widgets: **{total}**")

        # Latest evidences (last element of history)
        EHbD = st.session_state.EHb_list[-1]
        EHgD = st.session_state.EHg_list[-1]

        # Probabilities from evidences (same formulas as in Dash)
        PHbD = 1.0 / (1.0 + 10 ** (-EHbD / 10.0))
        PHgD = 1.0 / (1.0 + 10 ** (-EHgD / 10.0))

        # Display P(Hb|D) and P(Hg|D) with subscripts using Unicode
        st.markdown(f"**$P(H_b | D) = {PHbD:.2f}$**&nbsp; &nbsp; &nbsp;**$E(H_b|D) = {EHbD:.2f}$ dB**")
        st.markdown(f"**$P(H_g | D) = {PHgD:.2f}$**&nbsp; &nbsp; &nbsp;**$E(H_g|D) = {EHgD:.2f}$ dB**")
    with col_right:
        st.subheader("Evidences over time")

        # Build the Plotly figure to match the Dash graph
        x_vals = list(range(len(st.session_state.EHb_list)))

        trace_Hg = go.Scatter(
            x=x_vals,
            y=st.session_state.EHg_list,
            mode="lines+markers",
            line=dict(color="green"),
            name="E(Hg|D)",
        )
        trace_Hb = go.Scatter(
            x=x_vals,
            y=st.session_state.EHb_list,
            mode="lines+markers",
            line=dict(color="orange"),
            name="E(Hb|D)",
        )

        layout = go.Layout(
            title="Evidences for Hb and Hg",
            xaxis={"title": "Number of widgets tested"},
            yaxis={"title": "Evidences (dB)"},
            legend={"x": 0, "y": 1},
        )

        fig = go.Figure(data=[trace_Hb, trace_Hg], layout=layout)
        st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()


# import json

# import dash
# from dash import dcc, html, Input, Output, State
# import plotly.graph_objs as go
# import numpy as np

# app = dash.Dash(__name__)


# def eh(n_good, n_bad):
#     EH0 = -10 + 20.0 * np.log10(2) * n_bad + 10.0 * n_good * np.log10(2.0 / 5.0)
#     EH1 = 10 - 20.0 * np.log10(2) * n_bad + 10.0 * n_good * np.log10(5.0 / 2.0)
#     return EH0, EH1


# # Define the app layout
# app.layout = html.Div(
#     style={"display": "flex"},
#     children=[
#         # Div for buttons and counts, occupying 1/4 of the page
#         html.Div(
#             style={"flex": "1", "padding": "10px"},
#             children=[
#                 html.Button("Good Widget", id="button-good", n_clicks=0),
#                 html.Button("Bad Widget", id="button-bad", n_clicks=0),
#                 html.Div(
#                     id="count-data", style={"display": "none"}
#                 ),  # Hidden div to store counts
#                 html.Div(id="display-good"),
#                 html.Div(id="display-bad"),
#                 html.Div(id="display-total"),
#                 html.P(id="display-P-H0-D"),
#                 html.P(id="display-P-H1-D"),
#             ],
#         ),
#         # Div for the graph, occupying the rest of the page
#         html.Div(
#             style={"flex": "3", "padding": "10px"},
#             children=[
#                 dcc.Graph(id="live-update-graph"),
#             ],
#         ),
#     ],
# )


# @app.callback(
#     [
#         Output("live-update-graph", "figure"),
#         Output("count-data", "children"),
#         Output("display-good", "children"),
#         Output("display-bad", "children"),
#         Output("display-total", "children"),
#         Output("display-P-H0-D", "children"),
#         Output("display-P-H1-D", "children"),
#     ],
#     [Input("button-good", "n_clicks"), Input("button-bad", "n_clicks")],
#     [State("count-data", "children")],
# )
# def update_graph(n_good, n_bad, counts):
#     # If no data is present, initialize it
#     if counts is None:
#         counts = json.dumps({"E(H_0|D)": [], "E(H_1|D)": []})
#         EH0 = -10
#         EH1 = 10

#     # Load the previous counts
#     counts_data = json.loads(counts)

#     # Calculate the new evidences
#     EH0D, EH1D = eh(n_good, n_bad)

#     # Append new data
#     counts_data["E(H_0|D)"].append(EH0D)
#     counts_data["E(H_1|D)"].append(EH1D)

#     # Create traces for the graph
#     trace_H_0 = go.Scatter(
#         x=list(range(0, len(counts_data["E(H_0|D)"]))),
#         y=counts_data["E(H_0|D)"],
#         mode="lines+markers",
#         name="E(H\u2080|D)",
#     )
#     trace_H_1 = go.Scatter(
#         x=list(range(0, len(counts_data["E(H_1|D)"]))),
#         y=counts_data["E(H_1|D)"],
#         mode="lines+markers",
#         name="E(H\u2081|D)",
#     )

#     layout = go.Layout(
#         title="Evidences for H\u2080 and H\u2081",
#         xaxis={"title": "Number of widgets testd"},
#         yaxis={"title": "Evidences (db)"},
#         legend={"x": 0, "y": 1},
#     )

#     # Save updated counts
#     updated_counts = json.dumps(counts_data)

#     # Calculate the total and the probabilities
#     total = n_good + n_bad
#     PH0D = 1.0 / (1.0 + 10 ** (-EH0D / 10))
#     PH1D = 1.0 / (1.0 + 10 ** (-EH1D / 10))

#     # These are the strings that will be displayed on the webpage, with subscripts
#     display_strings = {
#         "display-good": f"Good widgets: {n_good}",
#         "display-bad": f"Bad widgets: {n_bad}",
#         "display-total": f"Total widgets: {total}",
#         "display-P-H0-D": html.Span(
#             [
#                 html.Span("P(H", style={"vertical-align": "baseline"}),
#                 html.Sub("0"),
#                 html.Span("|D) = ", style={"vertical-align": "baseline"}),
#                 html.Span(f"{PH0D:.4f}", style={"vertical-align": "baseline"}),
#             ]
#         ),
#         "display-P-H1-D": html.Span(
#             [
#                 html.Span("P(H", style={"vertical-align": "baseline"}),
#                 html.Sub("1"),
#                 html.Span("|D) = ", style={"vertical-align": "baseline"}),
#                 html.Span(f"{PH1D:.4f}", style={"vertical-align": "baseline"}),
#             ]
#         ),
#     }

#     return (
#         {"data": [trace_H_0, trace_H_1], "layout": layout},
#         updated_counts,
#         display_strings["display-good"],
#         display_strings["display-bad"],
#         display_strings["display-total"],
#         display_strings["display-P-H0-D"],
#         display_strings["display-P-H1-D"],
#     )


# # Run the app
# if __name__ == "__main__":
#     app.run(debug=True, port=8051)
