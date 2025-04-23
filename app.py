import io
import base64
import pandas as pd
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

from src.preprocess_data import build_summary_features, build_fft_features
from src.modal_callbacks import register_modal_callbacks

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout
def create_layout():
    return dbc.Container([
        html.H2("Condition Monitoring of Hydraulic Systems"),
        html.P("Upload raw merged cycles CSV (`hydraulic_cycles.csv`) with sensor columns only:"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag & drop or click to upload']),
            style={
                'width':'100%','height':'60px','lineHeight':'60px',
                'borderWidth':'1px','borderStyle':'dashed',
                'borderRadius':'5px','textAlign':'center'
            }, multiple=False
        ),
        html.Br(),
        html.Div(id='feature-table'),
        html.Div([
            html.Label('Cycle index:'),
            dcc.Input(id='cycle-index', type='number', min=0, step=1)
        ]),
        dbc.Button('Predict Cycle', id='predict-btn', color='primary', disabled=True, className='mt-2'),
        dcc.Store(id='stored-features'),
        # Modal placeholder; callbacks registered separately
        dbc.Modal([
            dbc.ModalHeader(html.H4("Prediction for Selected Cycle")),
            dbc.ModalBody(html.Div(id='modal-body')),
            dbc.ModalFooter(dbc.Button('Close', id='close-modal', className='ml-auto'))
        ], id='results-modal', size='lg', is_open=False)
    ], fluid=True)

app.layout = create_layout()

# Enable predict button only when features stored and index provided
@app.callback(
    Output('predict-btn','disabled'),
    Input('stored-features','data'),
    Input('cycle-index','value')
)
def enable_predict(feats_json, idx):
    return not feats_json or idx is None

# Parse upload and engineer features
@app.callback(
    Output('stored-features','data'),
    Output('feature-table','children'),
    Input('upload-data','contents')
)
def parse_and_engineer(contents):
    if not contents:
        return dash.no_update, dash.no_update
    _, b64 = contents.split(',')
    df = pd.read_csv(io.StringIO(base64.b64decode(b64).decode('utf-8')))
    # Build features
    sumf = build_summary_features(df)
    fftf = build_fft_features(df, n_bins=5)
    feats = pd.concat([sumf, fftf], axis=1)
    # Insert cycle index column
    feats.insert(0, 'cycle_index', feats.index)
    feats_json = feats.to_json(date_format='iso', orient='split')
    # Preview first 5 rows
    table = dash_table.DataTable(
        data=feats.head(5).to_dict('records'),
        columns=[{'name':c,'id':c} for c in feats.columns],
        page_size=5,
        style_table={'overflowX':'auto'}
    )
    return feats_json, html.Div([html.H5('Feature Preview (first 5 cycles)'), table])

# Register modal callbacks from external module
register_modal_callbacks(app)

if __name__=='__main__':
    app.run(debug=True)
