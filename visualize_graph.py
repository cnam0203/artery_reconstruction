import plotly.graph_objs as go


def show_figure(data):
    """
    The function `show_figure` in Python creates a 3D figure with specified layout settings and displays
    it.
    
    :param data: The `data` parameter in the `show_figure` function is typically a list of traces that
    define the data to be plotted in the figure. Each trace represents a different set of data points or
    a different type of visualization (e.g., scatter plot, line plot, bar chart)
    """
    layout = go.Layout(
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1, y=1, z=1)
            )
        ),
        height=1200,  # Set height to 800 pixels
        width=2000   # Set width to 1200 pixels
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(showlegend=False)
    fig.show()


def generate_points(points, point_size=2, point_color='black'):
    """
    The function `generate_points` creates a 3D scatter plot of points with specified size and color.
    
    :param points: The `points` parameter in the `generate_points` function is expected to be a 2D NumPy
    array containing the coordinates of the points in a 3D space. Each row of the array represents a
    point, and the columns represent the x, y, and z coordinates of that point
    :param point_size: The `point_size` parameter in the `generate_points` function is used to specify
    the size of the markers representing the points in a 3D scatter plot. This parameter allows you to
    control the visual appearance of the points by setting their size. The default value for
    `point_size` is, defaults to 2 (optional)
    :param point_color: The `point_color` parameter in the `generate_points` function is used to specify
    the color of the points in the 3D scatter plot. You can provide a color name (e.g., 'red', 'blue',
    'green') or a hexadecimal color code (e.g., '#FF, defaults to black (optional)
    :return: The function `generate_points` is returning a Scatter3d plot with the specified points,
    point size, and point color. The plot will display the points in a 3D space as markers with the
    given size and color.
    """

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=point_color
        ),
        name='Points',
        text=[f'Pos {i}' for i in range(points.shape[0])]
    )

def generate_points_viridis(points, point_size=2, point_values=None):
    """
    The function `generate_points` creates a 3D scatter plot of points with specified size and color.
    
    :param points: The `points` parameter in the `generate_points` function is expected to be a 2D NumPy
    array containing the coordinates of the points in a 3D space. Each row of the array represents a
    point, and the columns represent the x, y, and z coordinates of that point
    :param point_size: The `point_size` parameter in the `generate_points` function is used to specify
    the size of the markers representing the points in a 3D scatter plot. This parameter allows you to
    control the visual appearance of the points by setting their size. The default value for
    `point_size` is, defaults to 2 (optional)
    :param point_color: The `point_color` parameter in the `generate_points` function is used to specify
    the color of the points in the 3D scatter plot. You can provide a color name (e.g., 'red', 'blue',
    'green') or a hexadecimal color code (e.g., '#FF, defaults to black (optional)
    :return: The function `generate_points` is returning a Scatter3d plot with the specified points,
    point size, and point color. The plot will display the points in a 3D space as markers with the
    given size and color.
    """

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=point_values,  # Assign the array of values to color
            colorscale='Viridis',  # Specify the color scale
        ),
        name='Points',
        text=[f'Pos {i}' for i in range(points.shape[0])]
    )
    
def generate_lines(lines, line_width=2, line_color='black'):
    """
    The function `generate_lines` creates a 3D scatter plot of lines with specified width and color.
    
    :param lines: The `lines` parameter in the `generate_lines` function is expected to be a 2D NumPy
    array representing the coordinates of the lines in 3D space. Each row in the array should contain
    the x, y, and z coordinates of a point on the line
    :param line_width: The `line_width` parameter in the `generate_lines` function specifies the width
    of the lines that will be plotted in the 3D scatter plot. This parameter allows you to control the
    thickness of the lines that connect the points in the scatter plot. By adjusting the `line_width`
    value,, defaults to 2 (optional)
    :param line_color: The `line_color` parameter in the `generate_lines` function is used to specify
    the color of the lines that will be plotted in the 3D scatter plot. You can provide a color name
    (e.g., 'red', 'blue', 'green') or a hexadecimal color code (e, defaults to black (optional)
    :return: The function `generate_lines` is returning a `go.Scatter3d` object that represents a 3D
    line plot. The lines are defined by the input `lines` array, with specified line width and color.
    """
    
    return go.Scatter3d(
        x=lines[:, 0],
        y=lines[:, 1],
        z=lines[:, 2],
        mode='lines',
        line=dict(
            width=line_width,
            color=line_color
        ),
    )
