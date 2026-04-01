import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from matplotlib.patches import Ellipse

def extract_embeddings(model, data_segments, label, device):
    """
    Pass a list of time-series segments through the encoder to extract embeddings.

    Args:
        model (nn.Module): Model with an `embed()` method returning embeddings.
        data_segments (list of arrays): 1D segments to embed.
        label (str or int): Label to associate with all embeddings.
        device (torch.device): Device to run inference on.

    Returns:
        list of (embedding, label): Each embedding is a 1D NumPy array.
    """
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for segment in data_segments:
            x = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)
            embedding = model.embed(x).cpu().numpy().squeeze()
            all_embeddings.append((embedding, label))
    return all_embeddings


def plot_interactive_embeddings(embeddings, regions, electrodes, sides,
                                dim=3, method='umap', metric='euclidean', show_ellipses=True, verbose=True):
    """
    Interactive 2D or 3D plot of neural embeddings with color and shape annotations.

    Args:
        embeddings (list of array-like): Embedding vectors.
        regions (list of str): Region labels (e.g. "GPi1", "STN").
        electrodes (list of int): Electrode number identifiers.
        sides (list of str): Laterality ('L' or 'R').
        dim (int): Target dimensionality (2 or 3).
        method (str): Dimensionality reduction technique ('pca', 'tsne', or 'umap').
        metric (str): Distance metric (e.g., 'euclidean', 'cosine').

    Returns:
        None: Displays an interactive Plotly figure.
    """
    assert len(embeddings) == len(regions) == len(electrodes) == len(sides), "Length mismatch"
    assert method in ['pca', 'tsne', 'umap'], "Invalid method"
    assert dim in [2, 3], "dim must be 2 or 3"
    embeddings = np.array(embeddings)

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=dim)
    elif method == 'tsne':
        reducer = TSNE(n_components=dim, metric=metric, init='pca', random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=dim, metric=metric, random_state=42)

    reduced = reducer.fit_transform(embeddings)

    # Predefined region colors and marker shapes
    region_colors = {
        "GPi": "red", "GPi1": "red", "GPi2": "firebrick", "STN": "blue", "VO": "green", "VoaVop": "green", "Vo": "green","VOSTN": "blue", "VIMPPM": "purple",
        "VIM": "purple", "VPLa": "orange", "PPN": "teal","VA":"brown"
    }
    side_shapes = {"L": "circle", "R": "square"}

    plot_colors = [
    'blue',
    'red',
    'green',
    'orange',
    'purple',
    'cyan',
    'magenta',
    'yellow',
    'black',
    'brown',
    '#1f77b4',  # muted blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    ]
    # Organize into DataFrame
    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "z": reduced[:, 2] if dim == 3 else 0,
        "Region": regions,
        "Electrode": electrodes,
        "Side": sides
    })
    # df["Label"] = df["Region"] + "_E" + df["Electrode"].astype(str) + "_" + df["Side"]
    # df["Label"] = df["Region"] + "_" + df["Side"]
    df["Label"] = df["Region"]

    # Plot each electrode as a trace
    traces = []
    for label in df["Label"].unique():
        sub = df[df["Label"] == label]
        region = sub["Region"].iloc[0]
        side = sub["Side"].iloc[0]
        color = region_colors.get(region, plot_colors.pop(0))
        symbol = side_shapes.get(side, "circle")

        if dim == 3:
            trace = go.Scatter3d(
                x=sub["x"], y=sub["y"], z=sub["z"],
                mode='markers',
                name=label,
                hovertext=sub["Label"],
                marker=dict(size=6, color=color, symbol=symbol, opacity= 0.5)
            )
        else:
            trace = go.Scatter(
                x=sub["x"], y=sub["y"],
                mode='markers',
                name=label,
                hovertext=sub["Label"],
                marker=dict(size=6, color=color, symbol=symbol, opacity= 0.5)
            )
        traces.append(trace)

    # Add ellipses per region (2D only)
    if dim == 2 and show_ellipses and method != 'tsne':
        for region in df["Region"].unique():
            region_points = df[df["Region"] == region][["x", "y"]].values
            if len(region_points) < 5:
                continue  # skip small regions
            cov = np.cov(region_points, rowvar=False)
            mean = np.mean(region_points, axis=0)

            # Eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

            width, height = 2 * np.sqrt(eigvals)
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle)

            # Convert ellipse to polygon for Plotly
            t = np.linspace(0, 2 * np.pi, 100)
            ellipse_x = mean[0] + width/2 * np.cos(t) * np.cos(np.radians(angle)) - height/2 * np.sin(t) * np.sin(np.radians(angle))
            ellipse_y = mean[1] + width/2 * np.cos(t) * np.sin(np.radians(angle)) + height/2 * np.sin(t) * np.cos(np.radians(angle))

            ellipse_trace = go.Scatter(
                x=ellipse_x, y=ellipse_y,
                mode='lines',
                name=f"{region} Ellipse",
                line=dict(color=region_colors.get(region, "gray"), dash='dot'),
                showlegend=False
            )
            traces.append(ellipse_trace)

    title = f"{method.upper()} Embedding ({dim}D, metric={metric})"

    if dim ==3:
        layout = go.Layout(
            title=title,
            width=950,
            height=700,
            paper_bgcolor='rgb(255,255,255)', 
            plot_bgcolor='rgb(0,0,0)',    
            scene=dict(
                xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3",
                xaxis=dict(showbackground=False, showgrid=True, zeroline=False, nticks=5, color='black', gridcolor='gray', gridwidth=0.5, linecolor="black", linewidth=3,ticks="outside"),
                yaxis=dict(showbackground=False, showgrid=True, zeroline=False, nticks=5, color='black', gridcolor='gray', gridwidth=0.5, linecolor="black", linewidth=3,ticks="outside"),
                zaxis=dict(showbackground=False, showgrid=True, zeroline=False, nticks=5, color='black', gridcolor='gray', gridwidth=0.5, linecolor="black", linewidth=3,ticks="outside"),
                xaxis_backgroundcolor= "rgb(0.9,0.9,0.9)",
                yaxis_backgroundcolor= "rgb(0.9,0.9,0.9)",
                zaxis_backgroundcolor= "rgb(0.9,0.9,0.9)",
                bgcolor='rgba(0,0,0,0)'
            ),
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(x=0.8, y=0.5, xanchor='left', yanchor='top'),
            font=dict( size=14, color="black")
        )
    elif dim ==2:
        layout = go.Layout(
            title=title,
            width=950,
            height=700,
            paper_bgcolor='rgb(255,255,255)', 
            plot_bgcolor='rgb(255,255,255)',  
            xaxis_title="Dim 1", yaxis_title="Dim 2",  
            xaxis=dict(showgrid=True, zeroline=False, nticks=5, color='black', gridcolor='gray', gridwidth=0.5, griddash="dot", linecolor="black", linewidth=1,ticks="outside"),
            yaxis=dict(showgrid=True, zeroline=False, nticks=5, color='black', gridcolor='gray', gridwidth=0.5, griddash="dot", linecolor="black", linewidth=1,ticks="outside"),
            # margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(x=1, y=1, xanchor='left', yanchor='top'),
            font=dict( size=14, color="black")
        )

    
    

    fig = go.Figure(data=traces, layout=layout)
    if verbose: fig.show()
    
    return fig, reducer