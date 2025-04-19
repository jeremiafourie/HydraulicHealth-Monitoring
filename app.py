import os
import io
import base64
import pandas as pd
import joblib
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Paths and model definitions
ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'
MODEL_FILES = {
    'cooler_pct':   ARTIFACTS_DIR / 'rf_cooler_pct.pkl',
    'valve_pct':    ARTIFACTS_DIR / 'rf_valve_pct.pkl',
    'pump_leak':    ARTIFACTS_DIR / 'rf_pump_leak.pkl',
    'acc_pressure': ARTIFACTS_DIR / 'rf_acc_pressure.pkl'
}

# Path to a sample features CSV to extract expected column names
TEMPLATE_CSV = Path(__file__).parent / 'data' / 'processed' / 'features.csv'
try:
    expected_cols = pd.read_csv(TEMPLATE_CSV, nrows=0).columns.tolist()
except Exception:
    expected_cols = []

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Build list of expected columns UI
if expected_cols:
    expected_list = html.Ul([html.Li(col) for col in expected_cols], style={'columns': 2})
else:
    expected_list = html.P("No template CSV found to list expected columns.")

# Layout
app.layout = dbc.Container([
    html.H1("Hydraulic System Condition Monitor"),
    html.H5("Upload a CSV with the feature columns below:"),
    expected_list,

    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'marginBottom': '20px'
        },
        multiple=False
    ),

    html.Div([
        html.Label("Cycle index (optional):"),
        dcc.Input(id='cycle-index', type='number', min=0, step=1,
                  placeholder='Leave blank to predict all rows')
    ], style={'width': '300px', 'marginBottom': '20px'}),

    dbc.Button("Run Predictions", id='predict-btn', color='primary', disabled=True),

    dcc.Store(id='stored-data'),

    dbc.Modal([
        dbc.ModalHeader(html.H4("Prediction Results")),
        dbc.ModalBody(id='modal-body'),
        dbc.ModalFooter(dbc.Button("Close", id='close-modal'))
    ], id='results-modal', size='lg', is_open=False)
], fluid=True)

# Parse upload and enable button
@app.callback(
    Output('stored-data', 'data'),
    Output('predict-btn', 'disabled'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def parse_upload(contents, filename):
    if contents is None:
        return dash.no_update, True
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df.to_json(date_format='iso', orient='split'), False

# Run predictions and show modal
@app.callback(
    Output('results-modal', 'is_open'),
    Output('modal-body', 'children'),
    Input('predict-btn', 'n_clicks'),
    Input('close-modal', 'n_clicks'),
    State('cycle-index', 'value'),
    State('stored-data', 'data'),
    State('results-modal', 'is_open')
)
def show_results(run_click, close_click, idx, data_json, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, dash.no_update
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'close-modal':
        return False, dash.no_update

    # Load DataFrame
    df = pd.read_json(data_json, orient='split')
    # Select one row if index provided
    if idx is not None and 0 <= idx < len(df):
        df = df.iloc[[int(idx)]]

    bars = []
    for target, model_path in MODEL_FILES.items():
        model = joblib.load(model_path)
        preds = model.predict(df)
        counts = pd.Series(preds).value_counts(normalize=True)
        segments = [
            {'label': str(int(cls)), 'value': int(frac * 100), 'color': 'info'}
            for cls, frac in counts.items()
        ]
        bars.append(
            dbc.Row([
                dbc.Col(html.Strong(target), width='auto'),
                dbc.Col(dbc.Progress(segments, multi=True, style={'height': '40px'}), width=10)
            ], align='center', className='mb-3')
        )

    return True, bars

if __name__ == '__main__':
    app.run(debug=True)
