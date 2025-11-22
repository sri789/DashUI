from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np

# Initialize app with Bootstrap for nicer UI
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Hackathon Dashboard"

# Layout with Tabs
app.layout = dbc.Container([
    html.H1("Hackathon Dashboard - Dash UI", className="text-center my-4"),
    dcc.Tabs(id="tabs", value="upload", children=[
        dcc.Tab(label="Data Upload", value="upload"),
        dcc.Tab(label="Model Inference", value="inference"),
        dcc.Tab(label="Evaluation", value="evaluation"),
    ]),
    html.Div(id="tabs-content")
], fluid=True)


# Callback to render tab content
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "upload":
        return dbc.Container([
            html.H3("Upload Your Dataset"),
            dcc.Upload(
                id="upload-data",
                children=html.Div(["Drag and Drop or ", html.A("Select a CSV File")]),
                style={
                    "width": "100%", "height": "60px", "lineHeight": "60px",
                    "borderWidth": "1px", "borderStyle": "dashed",
                    "borderRadius": "5px", "textAlign": "center"
                },
                multiple=False
            ),
            html.Div(id="output-data-upload")
        ])
    elif tab == "inference":
        return dbc.Container([
            html.H3("Model Inference"),
            html.P("Run a simple demo model on uploaded data."),
            dbc.Button("Run Model", id="run-model", color="primary"),
            html.Div(id="model-output")
        ])
    elif tab == "evaluation":
        return dbc.Container([
            html.H3("Evaluation Metrics"),
            html.P("Placeholder accuracy and confusion matrix."),
            dcc.Graph(
                figure=px.imshow(
                    [[50, 10], [5, 35]],
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Positive", "Negative"], y=["Positive", "Negative"],
                    title="Confusion Matrix"
                )
            ),
            html.Div("Accuracy: 85% | F1 Score: 0.82")
        ])


# Handle file upload
@app.callback(Output("output-data-upload", "children"),
              Input("upload-data", "contents"),
              State("upload-data", "filename"))
def update_output(contents, filename):
    if contents is not None:
        # For demo: generate random data instead of parsing file
        df = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})
        fig = px.scatter(df, x="x", y="y", title=f"Preview of {filename}")
        return dcc.Graph(figure=fig)
    return html.Div("No file uploaded yet.")


# Handle model inference
@app.callback(Output("model-output", "children"),
              Input("run-model", "n_clicks"))
def run_model(n_clicks):
    if n_clicks:
        # Demo: pretend model predicts random values
        preds = np.random.choice(["Positive", "Negative"], size=10)
        return html.Div([
            html.P("Model Predictions:"),
            html.Ul([html.Li(p) for p in preds])
        ])
    return html.Div("Click 'Run Model' to start inference.")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)