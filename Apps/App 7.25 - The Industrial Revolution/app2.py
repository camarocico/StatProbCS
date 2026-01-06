import math
from typing import Tuple

import numpy as np
import plotly.graph_objs as go
import streamlit as st
from scipy.special import comb


# --------- Core computations (same logic as Dash) ---------


def compute_priors() -> Tuple[float, float, float]:
    """
    Returns (PHb, PHg, PHr) as in the Dash app:
      PHr = 1e-6
      PHb = (1 - PHr) * 1/10
      PHg = (1 - PHr) * 9/10
    """
    PHr = 1.0 / 1_000_000.0
    PHb = (1.0 - PHr) * 1.0 / 10.0
    PHg = (1.0 - PHr) * 9.0 / 10.0
    return PHb, PHg, PHr


def compute_prior_evidence(PHb: float, PHg: float, PHr: float) -> Tuple[float, float, float]:
    """
    Evidences in dB from prior odds: E = 10 * log10(O),
    O = P / (1 - P).
    """
    OHb = PHb / (1.0 - PHb)
    OHg = PHg / (1.0 - PHg)
    OHr = PHr / (1.0 - PHr)

    EHb = 10.0 * np.log10(OHb)
    EHg = 10.0 * np.log10(OHg)
    EHr = 10.0 * np.log10(OHr)

    return EHb, EHg, EHr

def compute_likelihoods(n_good: int, n_bad: int) -> Tuple[float, float, float]:
    """
    PDHk = P(D | H_k) for k = b,g,r,
    with the same binomial models as the Dash code.
    """
    n = n_good + n_bad
    c = comb(n, n_bad)

    PDHb = c * (1.0 / 3.0) ** n_good * (2.0 / 3.0) ** n_bad
    PDHg = c * (5.0 / 6.0) ** n_good * (1.0 / 6.0) ** n_bad
    PDHr = c * (1.0 / 20.0) ** n_good * (19.0 / 20.0) ** n_bad

    return PDHb, PDHg, PDHr

def compute_B_factors(
    PHb: float, PHg: float, PHr: float,
    PDHb: float, PDHg: float, PDHr: float,
) -> Tuple[float, float, float]:
    """
    BbD, BgD, BrD as in the Dash app.
    """
    BbD = PDHb * (PHg + PHr) / (PHg * PDHg + PHr * PDHr)
    BgD = PDHg * (PHb + PHr) / (PHb * PDHb + PHr * PDHr)
    BrD = PDHr * (PHb + PHg) / (PHb * PDHb + PHg * PDHg)
    return BbD, BgD, BrD


def compute_posterior_evidence(
    n_good: int,
    n_bad: int,
) -> Tuple[float, float, float, float, float, float]:
    """
    For given counts (n_good, n_bad), compute:
      EHbD, EHgD, EHrD (posterior evidence in dB)
      PHbD, PHgD, PHrD (posterior probabilities)
    using exactly the same formulas as the Dash callback.
    """
    PHb, PHg, PHr = compute_priors()
    EHb, EHg, EHr = compute_prior_evidence(PHb, PHg, PHr)
    PDHb, PDHg, PDHr = compute_likelihoods(n_good, n_bad)
    BbD, BgD, BrD = compute_B_factors(PHb, PHg, PHr, PDHb, PDHg, PDHr)

    EHbD = EHb + 10.0 * np.log10(BbD)
    EHgD = EHg + 10.0 * np.log10(BgD)
    EHrD = EHr + 10.0 * np.log10(BrD)

    # Convert evidence (in dB) back to probabilities:
    # P = 1 / (1 + 10^(-E/10))
    PHbD = 1.0 / (1.0 + 10 ** (-EHbD / 10.0))
    PHgD = 1.0 / (1.0 + 10 ** (-EHgD / 10.0))
    PHrD = 1.0 / (1.0 + 10 ** (-EHrD / 10.0))

    return EHbD, EHgD, EHrD, PHbD, PHgD, PHrD

# --------- State management ---------


def init_state():
    """
    Initialize Streamlit session_state with:
      n_good, n_bad, and history lists of evidences for Hb, Hg, Hr.
    We seed the history with the 'zero data' point (n_good = n_bad = 0),
    mirroring the Dash app's first callback.
    """
    if "n_good" not in st.session_state:
        st.session_state.n_good = 0
    if "n_bad" not in st.session_state:
        st.session_state.n_bad = 0
    if "EHb_list" not in st.session_state or "EHg_list" not in st.session_state or "EHr_list" not in st.session_state:
        EHbD, EHgD, EHrD, _, _, _ = compute_posterior_evidence(0, 0)
        st.session_state.EHb_list = [EHbD]
        st.session_state.EHg_list = [EHgD]
        st.session_state.EHr_list = [EHrD]

# --------- Streamlit app ---------


def main():
    st.set_page_config(
        page_title="Evidences for Hb, Hg and Hr",
        layout="wide",
    )

    init_state()

    st.title("Evidences for $H_b$, $H_g$ and $H_r$")

    col_left, col_right = st.columns([2, 5])

    with col_left:
        st.subheader("Widget inspection")

        clicked_good = st.button("Good Widget")
        clicked_bad = st.button("Bad Widget")

        # Update counts and evidence history when a button is clicked
        if clicked_good:
            st.session_state.n_good += 1
        if clicked_bad:
            st.session_state.n_bad += 1

        if clicked_good or clicked_bad:
            EHbD, EHgD, EHrD, _, _, _ = compute_posterior_evidence(
                st.session_state.n_good, st.session_state.n_bad
            )
            st.session_state.EHb_list.append(EHbD)
            st.session_state.EHg_list.append(EHgD)
            st.session_state.EHr_list.append(EHrD)

        n_good = st.session_state.n_good
        n_bad = st.session_state.n_bad
        total = n_good + n_bad

        st.write(f"Good widgets: **{n_good}**")
        st.write(f"Bad widgets: **{n_bad}**")
        st.write(f"Total widgets: **{total}**")

        # Current evidences and posterior probabilities
        EHbD = st.session_state.EHb_list[-1]
        EHgD = st.session_state.EHg_list[-1]
        EHrD = st.session_state.EHr_list[-1]

        # Recompute posteriors from the evidences (same mapping as in Dash)
        PHbD = 1.0 / (1.0 + 10 ** (-EHbD / 10.0))
        PHgD = 1.0 / (1.0 + 10 ** (-EHgD / 10.0))
        PHrD = 1.0 / (1.0 + 10 ** (-EHrD / 10.0))

        st.markdown(f"**$P(H_b | D) = {PHbD:.2f}$**&nbsp; &nbsp; &nbsp;**$E(H_b|D) = {EHbD:.2f}$ dB**")
        st.markdown(f"**$P(H_g | D) = {PHgD:.2f}$**&nbsp; &nbsp; &nbsp;**$E(H_g|D) = {EHgD:.2f}$ dB**")
        st.markdown(f"**$P(H_r | D) = {PHrD:.2f}$**&nbsp; &nbsp; &nbsp;**$E(H_r|D) = {EHrD:.2f}$ dB**")

    with col_right:
        st.subheader("Evidences over time")

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
        trace_Hr = go.Scatter(
            x=x_vals,
            y=st.session_state.EHr_list,
            mode="lines+markers",
            line=dict(color="red"),
            name="E(Hr|D)",
        )


        layout = go.Layout(
            title="Evidences for Hb, Hg and Hr",
            xaxis={"title": "Number of widgets tested"},
            yaxis={"title": "Evidences (dB)"},
            legend={"x": 0, "y": 1},
        )

        fig = go.Figure(data=[trace_Hb, trace_Hg, trace_Hr], layout=layout)
        st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    main()

# import json

# import dash
# from dash import dcc, html, Input, Output, State
# import plotly.graph_objs as go
# import numpy as np
# from scipy.special import comb

# app = dash.Dash(__name__)


# def eh(EH0, EH1, EH2, B0D, B1D, B2D, n_good, n_bad):
#     EH0D = EH0 + 10.0 * np.log10(B0D)
#     EH1D = EH1 + 10.0 * np.log10(B1D)
#     EH2D = EH2 + 10.0 * np.log10(B2D)
#     print(EH0D, EH1D, EH2D)
#     return EH0D, EH1D, EH2D


# # Define the app layout
# app.layout = html.Div(style={'display': 'flex'}, children=[
#     # Div for buttons and counts, occupying 1/4 of the page
#     html.Div(style={'flex': '1', 'padding': '10px'}, children=[
#         html.Button('Good Widget', id='button-good', n_clicks=0),
#         html.Button('Bad Widget', id='button-bad', n_clicks=0),
#         html.Div(id='count-data', style={'display': 'none'}),
#         # Hidden div to store counts
#         html.Div(id='display-good'),
#         html.Div(id='display-bad'),
#         html.Div(id='display-total'),
#         html.P(id='display-P-H0-D'),
#         html.P(id='display-P-H1-D'),
#         html.P(id='display-P-H2-D')
#     ]),

#     # Div for the graph, occupying the rest of the page
#     html.Div(style={'flex': '3', 'padding': '10px'}, children=[
#         dcc.Graph(id='live-update-graph'),
#     ]),
# ])


# @app.callback(
#     [Output('live-update-graph', 'figure'),
#      Output('count-data', 'children'),
#      Output('display-good', 'children'),
#      Output('display-bad', 'children'),
#      Output('display-total', 'children'),
#      Output('display-P-H0-D', 'children'),
#      Output('display-P-H1-D', 'children'),
#      Output('display-P-H2-D', 'children')],
#     [Input('button-good', 'n_clicks'),
#      Input('button-bad', 'n_clicks')],
#     [State('count-data', 'children')]
# )
# def update_graph(n_good, n_bad, counts):
#     PH2 = 1 / 1000000
#     PH0 = (1 - PH2) * 1 / 10
#     PH1 = (1 - PH2) * 9 / 10
#     OH0 = PH0 / (1 - PH0)
#     OH1 = PH1 / (1 - PH1)
#     OH2 = PH2 / (1 - PH2)
#     EH0 = 10 * np.log10(OH0)
#     EH1 = 10 * np.log10(OH1)
#     EH2 = 10 * np.log10(OH2)

#     PDH0 = comb(n_good + n_bad, n_bad) * (1.0 / 3.0) ** n_good * (2.0 / 3.0) ** n_bad
#     PDH1 = comb(n_good + n_bad, n_bad) * (5.0 / 6.0) ** n_good * (1.0 / 6.0) ** n_bad
#     PDH2 = comb(n_good + n_bad, n_bad) * (1.0 / 20.0) ** n_good * (19.0 / 20.0) ** n_bad

#     B0D = PDH0 * (PH1 + PH2) / (PH1 * PDH1 + PH2 * PDH2)
#     B1D = PDH1 * (PH0 + PH2) / (PH0 * PDH0 + PH2 * PDH2)
#     B2D = PDH2 * (PH0 + PH1) / (PH0 * PDH0 + PH1 * PDH1)

#     # If no data is present, initialize it
#     if counts is None:
#         counts = json.dumps({'E(H_0|D)': [], 'E(H_1|D)': [], 'E(H_2|D)': []})

#     # Load the previous counts
#     counts_data = json.loads(counts)

#     # Calculate the new evidences
#     EH0D, EH1D, EH2D = eh(EH0, EH1, EH2, B0D, B1D, B2D, n_good, n_bad)

#     # Append new data
#     counts_data['E(H_0|D)'].append(EH0D)
#     counts_data['E(H_1|D)'].append(EH1D)
#     counts_data['E(H_2|D)'].append(EH2D)

#     # Create traces for the graph
#     trace_H_0 = go.Scatter(
#         x=list(range(0, len(counts_data['E(H_0|D)']))), y=counts_data['E(H_0|D)'],
#         mode='lines+markers', name='E(H\u2080|D)'
#     )
#     trace_H_1 = go.Scatter(
#         x=list(range(0, len(counts_data['E(H_1|D)']))), y=counts_data['E(H_1|D)'],
#         mode='lines+markers', name='E(H\u2081|D)'
#     )
#     trace_H_2 = go.Scatter(
#         x=list(range(0, len(counts_data['E(H_2|D)']))), y=counts_data['E(H_2|D)'],
#         mode='lines+markers', name='E(H\u2082|D)'
#     )

#     layout = go.Layout(title='Evidences for H\u2080, H\u2081, and H\u2082',
#                        xaxis={'title': 'Number of widgets tested'},
#                        yaxis={'title': 'Evidences (db)'},
#                        legend={'x': 0, 'y': 1})

#     # Save updated counts
#     updated_counts = json.dumps(counts_data)

#     # Calculate the total and the probabilities
#     total = n_good + n_bad
#     PH0D = 1.0 / (1.0 + 10**(-EH0D / 10))
#     PH1D = 1.0 / (1.0 + 10**(-EH1D / 10))
#     PH2D = 1.0 / (1.0 + 10**(-EH2D / 10))

#     # These are the strings that will be displayed on the webpage, with subscripts
#     display_strings = {
#         'display-good': f'Good widgets: {n_good}',
#         'display-bad': f'Bad widgets: {n_bad}',
#         'display-total': f'Total widgets: {total}',
#         'display-P-H0-D': html.Span(
#             [html.Span('P(H', style={'vertical-align': 'baseline'}),
#              html.Sub('0'),
#              html.Span('|D) = ', style={'vertical-align': 'baseline'}),
#              html.Span(f'{PH0D:.4f}', style={'vertical-align': 'baseline'})]),
#         'display-P-H1-D': html.Span(
#             [html.Span('P(H', style={'vertical-align': 'baseline'}),
#              html.Sub('1'),
#              html.Span('|D) = ', style={'vertical-align': 'baseline'}),
#              html.Span(f'{PH1D:.4f}', style={'vertical-align': 'baseline'})]),
#         'display-P-H2-D': html.Span(
#             [html.Span('P(H', style={'vertical-align': 'baseline'}),
#              html.Sub('2'),
#              html.Span('|D) = ', style={'vertical-align': 'baseline'}),
#              html.Span(f'{PH2D:.4f}', style={'vertical-align': 'baseline'})])
#     }

#     return ({'data': [trace_H_0, trace_H_1, trace_H_2], 'layout': layout}, updated_counts,
#             display_strings['display-good'],
#             display_strings['display-bad'],
#             display_strings['display-total'],
#             display_strings['display-P-H0-D'],
#             display_strings['display-P-H1-D'],
#             display_strings['display-P-H2-D'])


# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True, port=8052)
