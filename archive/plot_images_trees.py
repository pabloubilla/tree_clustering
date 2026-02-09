from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import pandas as pd
import os

# C:\Users\pablo\OneDrive\Desktop\tree_clustering\data\traits_pred_log.csv
# data_path = os.path.join('data', 'traits_pred_log.csv')
# take example
data_path = os.path.join('data', 'tree_plot_example.csv')

df = pd.read_csv(data_path,sep = ';', header=0, index_col=0)

fig = go.Figure(data=[
    go.Scatter(
        x=df["Leaf area"],
        y=df["Tree height"],
        mode="markers",
        marker=dict(
            # colorscale='viridis',
            # color=df["MW"],
            # size=df["MW"],
            # colorbar={"title": "Molecular<br>Weight"},
            # line={"color": "#444"},
            reversescale=True,
            sizeref=45,
            sizemode="diameter",
            opacity=0.8,
        )
    )
])

# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig.update_traces(hoverinfo="none", hovertemplate=None)

fig.update_layout(
    xaxis=dict(title= 'Tree height'),
    yaxis=dict(title= 'Leaf area'),
    plot_bgcolor='rgba(255,255,255,0.1)'
)

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
])


@callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src = df_row['IMG_URL']
    name = df_row['NAME']
    desc = df_row['DESC']
    if len(desc) > 300:
        desc = desc[:100] + '...'

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.H2(f"{name}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
            html.P(f"{desc}"),
        ], style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    # print(df)
    # # take a sample of 10 rows
    # df_sample = df.sample(10)
    # # save the sample to a csv file
    # df_sample.to_csv('data/tree_plot_example.csv')
    app.run(debug=True)
