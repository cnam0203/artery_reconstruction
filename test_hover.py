import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Sample data
lines = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 1, 2],
    [4, 4, 3],
    [5, 3, 5]
])
line_width = 5
line_color = 'blue'

# Generate scatter points data from lines
scatter_x = lines[:, 0]
scatter_y = lines[:, 1]
scatter_z = lines[:, 2]

# Layout of the app
app.layout = html.Div([
    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter3d(
                    x=lines[:, 0],
                    y=lines[:, 1],
                    z=lines[:, 2],
                    mode='lines',
                    line=dict(
                        width=line_width,
                        color=line_color
                    ),
                    name='Lines'
                ),
                go.Scatter3d(
                    x=scatter_x,
                    y=scatter_y,
                    z=scatter_z,
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    name='Hover Points',
                    visible=False  # Initially hide hover points
                )
            ],
            'layout': go.Layout(
                hovermode='closest'
            )
        }
    )
])

# Callback to show/hide the hover point
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-plot', 'hoverData')],
    [State('scatter-plot', 'figure')]
)
def display_hover_data(hoverData, figure):
    ctx = dash.callback_context
    if ctx.triggered:
        if hoverData:
            point_index = hoverData['points'][0]
            print(point_index)
            # figure['data'][1]['x'] = [scatter_x[point_index]]
            # figure['data'][1]['y'] = [scatter_y[point_index]]
            # figure['data'][1]['z'] = [scatter_z[point_index]]
            # figure['data'][1]['visible'] = True
        else:
            figure['data'][1]['visible'] = False
    return figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
