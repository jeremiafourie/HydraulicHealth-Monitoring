import io
import base64
import pandas as pd
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc

from src.preprocess_data import build_summary_features, build_fft_features
from src.modal_callbacks import register_modal_callbacks

# Initialize Dash app with a clean theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
server = app.server

# Navbar
navbar = dbc.NavbarSimple(
    brand="Hydraulic Condition Monitoring",
    color="dark",
    dark=True,
    fluid=True,
    className="mb-4"
)

# Upload Card
upload_card = dbc.Card(
    [
        dbc.CardHeader(html.H5("Upload Data")),
        dbc.CardBody(
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag & drop or click to upload CSV']),
                style={
                    'width': '100%',
                    'height': '80px',
                    'lineHeight': '80px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'backgroundColor': '#f9f9f9'
                },
                multiple=False
            )
        ),
    ],
    className="mb-4 shadow-sm"
)

# Cycle Selection & Predict Button Card
controls_card = dbc.Card(
    [
        dbc.CardHeader(html.H5("Cycle Prediction")),
        dbc.CardBody([
            html.Div([
                dbc.Label('Cycle Index', html_for='cycle-index', className='form-label'),
                dcc.Input(
                    id='cycle-index',
                    type='number',
                    min=0,
                    step=1,
                    placeholder='Enter cycle index',
                    className='form-control'
                ),
            ], className='mb-3'),
            dbc.Button(
                'Predict Cycle',
                id='predict-btn',
                color='primary',
                className='w-100',
                disabled=True
            )
        ])
    ],
    className="mb-4 shadow-sm"
)

# Feature Preview Card Placeholder
preview_card = dbc.Card(
    [
        dbc.CardHeader(html.H5('Feature Preview')),
        dbc.CardBody(
            dcc.Loading(
                id='loading-preview',
                type='circle',
                children=html.Div(id='feature-table')
            )
        )
    ],
    className="mb-4 shadow-sm"
)

# Layout
app.layout = dbc.Container([
    navbar,
    dbc.Row([
        dbc.Col([upload_card, controls_card], width=4),
        dbc.Col([preview_card], width=8)
    ], align='start'),

    # Error message display
    html.Div(id='error-message', className='mb-3'),

    # Hidden store for features
    dcc.Store(id='stored-features'),

    # Results Modal
    dbc.Modal([
        dbc.ModalHeader(html.H4("Prediction Result")),
        dbc.ModalBody(
            dcc.Loading(
                id='loading-prediction',
                type='circle',
                children=html.Div(id='modal-body')
            )
        ),
        dbc.ModalFooter(
            dbc.Button('Close', id='close-modal', className='ms-auto')
        ),
    ], id='results-modal', size='lg', is_open=False)

], fluid=True)

# Enable predict button only when features loaded and index provided
@app.callback(
    Output('predict-btn', 'disabled'),
    [Input('stored-features', 'data'), Input('cycle-index', 'value')]
)
def enable_predict(feats_json, idx):
    return feats_json is None or idx is None

# Parse upload and build features, catch CSV & feature errors
@app.callback(
    [Output('stored-features','data'), Output('feature-table','children')],
    [Input('upload-data','contents')]
)
def parse_and_engineer(contents):
    if not contents:
        return dash.no_update, dash.no_update
    # Try reading CSV
    try:
        _, b64 = contents.split(',')
        df = pd.read_csv(io.StringIO(base64.b64decode(b64).decode('utf-8')))
    except Exception:
        return None, html.Div(
            dbc.Alert("Error: Uploaded file is not a valid CSV.", color='danger'),
            className='p-2'
        )
    # Try building features
    try:
        sumf = build_summary_features(df)
        fftf = build_fft_features(df, n_bins=5)
        feats = pd.concat([sumf, fftf], axis=1)
        feats.insert(0, 'cycle_index', feats.index)
    except Exception as e:
        return None, html.Div(
            dbc.Alert(f"Error processing features: {str(e)}", color='danger'),
            className='p-2'
        )

    feats_json = feats.to_json(date_format='iso', orient='split')
    # Preview table
    table = dash_table.DataTable(
        data=feats.head(10).to_dict('records'),
        columns=[{'name': c, 'id': c} for c in feats.columns],
        page_size=8,
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
        ]
    )
    return feats_json, table

# Show errors on invalid cycle selection or CSV upload
@app.callback(
    Output('error-message', 'children'),
    [Input('upload-data','contents'), Input('predict-btn','n_clicks')],
    [State('stored-features','data'), State('cycle-index','value')]
)
def display_errors(contents, n_clicks, feats_json, idx):
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered == 'predict-btn':
        if feats_json is None:
            return dbc.Alert("Please upload a valid file and check feature errors.", color='warning')
        feats = pd.read_json(feats_json, orient='split')
        if idx is None or idx not in feats['cycle_index'].values:
            return dbc.Alert(f"Cycle index {idx} does not exist.", color='warning')
    return ''

# Register external modal callbacks
register_modal_callbacks(app)

if __name__ == '__main__':
    app.run(debug=False)
