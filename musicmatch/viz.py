from pathlib import Path
from jinja2 import Environment

import numpy as np

from musicmatch.config import DB_PATH, MPD_MUSIC_DIR


def get_all_file_embeddings(con):
    rows = (
        con.cursor()
        .execute(
            "SELECT filepath, embedding FROM track_chunks ORDER BY filepath, chunk_index"
        )
        .fetchall()
    )

    sums = {}
    counts = {}
    for fp, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        if fp not in sums:
            sums[fp] = emb.copy()
            counts[fp] = 1
        else:
            sums[fp] += emb
            counts[fp] += 1

    filepaths = sorted(sums.keys())
    embeddings = np.array([sums[fp] / counts[fp] for fp in filepaths], dtype=np.float32)
    return filepaths, embeddings


def pca(X, n_components=3):
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T


def _extract_group(filepath):
    music_dir = str(Path(MPD_MUSIC_DIR).expanduser().resolve())
    try:
        rel = str(Path(filepath).resolve().relative_to(music_dir))
    except ValueError:
        rel = filepath
    parts = rel.split("/")
    if len(parts) >= 3:
        return parts[0] + " / " + parts[1]
    elif len(parts) >= 2:
        return parts[0]
    return "Unknown"


_COLORS = [
    "#636efa",
    "#EF553B",
    "#00cc96",
    "#ab63fa",
    "#FFA15A",
    "#19d3f3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def scatter_3d(coords, filepaths, output_path, limit=None):
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError("plotly is required — run: uv pip install plotly")

    if limit and len(filepaths) > limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(filepaths), limit, replace=False)
        coords = coords[idx]
        filepaths = [filepaths[i] for i in idx]

    groups = [_extract_group(fp) for fp in filepaths]
    unique = list(dict.fromkeys(groups))
    group_map = {g: _COLORS[i % len(_COLORS)] for i, g in enumerate(unique)}
    colors = [group_map[g] for g in groups]

    fig = go.Figure(
        data=go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            text=filepaths,
            hoverinfo="text",
            hoverlabel=dict(bgcolor="#1a1a2e", font=dict(color="#e0e0e0", size=12)),
            marker=dict(
                size=4,
                color=colors,
                opacity=0.85,
            ),
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        scene=dict(
            xaxis=dict(
                showgrid=True, gridcolor="#2a2a4e", zeroline=False, visible=False
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#2a2a4e", zeroline=False, visible=False
            ),
            zaxis=dict(
                showgrid=True, gridcolor="#2a2a4e", zeroline=False, visible=False
            ),
            bgcolor="#1a1a2e",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        dragmode="orbit",
        hovermode="closest",
    )

    fig.update_layout(
        updatemenus=[],
        annotations=[],
    )

    plot_json = fig.to_json()

    template_path = Path(__file__).parent / "templates" / "visualization.html"
    env = Environment()
    html = (
        env.from_string(template_path.read_text(encoding="utf-8"))
        .render(
            plot_json=plot_json,
            filepaths=filepaths,
            colors=colors,
        )
    )

    Path(output_path).write_text(html, encoding="utf-8")
