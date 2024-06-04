import plotly.graph_objs as go


def show_figure(data, title="3D Mesh Figure"):
    """
    The function `show_figure` in Python creates a 3D figure with specified layout settings and displays
    it.
    
    :param data: The `data` parameter in the `show_figure` function is typically a list of traces that
    define the data to be plotted in the figure. Each trace represents a different set of data points or
    a different type of visualization (e.g., scatter plot, line plot, bar chart)
    """
    layout = go.Layout(
        title=title,
        scene=dict(
            aspectmode='manual',
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
            # camera=dict(
            #     eye=dict(x=1, y=1, z=1)
            # ),
            # aspectratio=dict(x=1, y=1, z=1)
        ),
        # height=1500,  # Set height to 800 pixels
        # width=1500   # Set width to 1200 pixels
    )

    fig = go.Figure(data=data, layout=layout)
    # fig.update_layout(showlegend=False)
    fig.update_layout(autosize=False, width=2000, height=1200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', )
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

def generate_mesh(vertices, faces):
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        # opacity=0.3,
        # color='#50ad61'
    )

    return mesh

def generate_mesh_color(vertices, faces, colors, title=''):
    hover_text = [f'Color: {color}' for color in colors]
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=colors,
        colorscale='Viridis',
        hoverinfo='text',
        hovertext=hover_text,
        name='Stenosis ratio',
        colorbar=dict(title='Stenosis Ratio')
    )

    return mesh

def generate_points_values(points, point_size=2, point_color='black', point_values=[]):
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
        text=[f'{point_values[i]}' for i in range(points.shape[0])]
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
