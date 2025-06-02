import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Load and clean data
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip().replace('\n', ' ') if isinstance(x, str) else x)

# Column order for the simplified path
columns = [
    "Feedstock Sources",
    "Energy  used  in  the process",
    "Process Type",
    "Fuel type"
]

# Color map for columns
column_colors = {
    "Feedstock Sources": "#2ca02c",
    "Energy  used  in  the process": "#9467bd",
    "Process Type": "#ff7f0e",
    "Fuel type": "#d62728"
}

# Dash app
app = Dash(__name__)
server = app.server
app.title = "Fuel Pathway Network"

# Layout
app.layout = html.Div([
    html.H2("Fuel Pathway Viewer"),

    html.Div([
        html.Label("Select Fuel Pathway Code"),
        dcc.Dropdown(
            options=[{"label": code, "value": code} for code in sorted(df["Fuel Pathway Code"].unique())],
            id="pathway-filter",
            multi=True
        ),
    ]),

    dcc.Graph(id="network-graph")
])

# Callback
@app.callback(
    Output("network-graph", "figure"),
    Input("pathway-filter", "value")
)
def update_graph(pathways):
    filtered = df.copy()
    if pathways:
        filtered = filtered[filtered["Fuel Pathway Code"].isin(pathways)]

    # Create subgraph
    subG = nx.DiGraph()
    for _, row in filtered.iterrows():
        for i in range(len(columns) - 1):
            src = row[columns[i]]
            tgt = row[columns[i + 1]]
            if pd.notna(src) and pd.notna(tgt):
                subG.add_edge(src, tgt)

    def get_horizontal_layout(graph, levels):
        pos = {}
        nodes_by_column = {level: set() for level in levels}
        for _, row in filtered.iterrows():
            for i, col in enumerate(levels):
                val = row[col]
                if pd.notna(val):
                    nodes_by_column[col].add(val)
        x_spacing = 300
        y_spacing = 50
        for i, col in enumerate(levels):
            col_nodes = sorted(nodes_by_column[col])
            for j, node in enumerate(col_nodes):
                pos[node] = (i * x_spacing, -j * y_spacing)
        return pos

    pos = get_horizontal_layout(subG, columns)

    
    edge_shapes = []
    for edge in subG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_shapes.append(
            dict(
                type="line",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="#888", width=1),
                opacity=1,
                axref='x', ayref='y',
                xref='x', yref='y',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1
            )
        )

    node_x, node_y, labels, node_colors = [], [], [], []
    for node in subG.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        for col in columns:
            if node in df[col].values:
                node_colors.append(column_colors[col])
                break

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            color=node_colors,
            size=20,
            line_width=2
        ),
        showlegend=False
    )

    # Legend
    legend_items = []
    for label, color in column_colors.items():
        legend_items.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=label,
            showlegend=True,
            name=label
        ))

    title = f"Fuel Pathway(s): {', '.join(pathways)}" if pathways else "Fuel Pathway Viewer"

    fig = go.Figure(data=[node_trace] + legend_items)
    fig.update_layout(
        shapes=edge_shapes,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

