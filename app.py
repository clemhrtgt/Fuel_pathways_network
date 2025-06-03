import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Load and clean data
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip().replace('\n', ' ') if isinstance(x, str) else x)

# Column order
columns = [
    "Feedstock Sources",
    "Energy  used  in  the process",
    "Process Type",
    "Fuel type"
]

column_colors = {
    "Feedstock Sources": "#2ca02c",
    "Energy  used  in  the process": "#9467bd",
    "Process Type": "#ff7f0e",
    "Fuel type": "#d62728"
}

category_emojis = {
    "Feedstock Sources": "🧺",
    "Energy  used  in  the process": "⚡",
    "Process Type": "⚙️",
    "Fuel type": "⛽"
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
            options=[
                {"label": f'{int(row["Order"])}: {row["Fuel Pathway Code"]}', "value": row["Fuel Pathway Code"]}
                for _, row in df[["Order", "Fuel Pathway Code"]].drop_duplicates().sort_values("Order").iterrows()
            ],
            id="pathway-filter",
            multi=True
        ),
    ]),

    dcc.Graph(id="network-graph", style={"height": "800px"})
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

    subG = nx.DiGraph()
    for _, row in filtered.iterrows():
        fs = row["Feedstock Sources"]
        en = row["Energy  used  in  the process"]
        pt = row["Process Type"]
        ft = row["Fuel type"]
        if pd.notna(fs) and pd.notna(pt):
            subG.add_edge(fs, pt)
        if pd.notna(en) and pd.notna(pt):
            subG.add_edge(en, pt)
        if pd.notna(pt) and pd.notna(ft):
            subG.add_edge(pt, ft)

    def get_custom_layout():
        pos = {}
        node_groups = {col: set() for col in columns}
        for _, row in filtered.iterrows():
            for col in columns:
                if pd.notna(row[col]):
                    node_groups[col].add(row[col])

        spacing = 300
        y_spacing = 60
        x_map = {
            "Feedstock Sources": 0,
            "Energy  used  in  the process": 1,
            "Process Type": 2,
            "Fuel type": 3
        }

        for col in columns:
            nodes = sorted(node_groups[col])
            for j, node in enumerate(nodes):
                x = x_map[col] * spacing
                y = -j * y_spacing
                pos[node] = (x, y)
        return pos

    pos = get_custom_layout()

    def create_arrow(x0, y0, x1, y1):
        return dict(
            ax=x0, ay=y0,
            x=x1, y=y1,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="black",
            opacity=1
        )

    annotations = []
    for edge in subG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        annotations.append(create_arrow(x0, y0, x1, y1))

    # Node emojis
    node_x, node_y, node_text, hover_texts = [], [], [], []
    for node in subG.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        hover_texts.append(node)

        emoji = "❓"
        for col in columns:
            if node in df[col].values:
                emoji = category_emojis[col]
                break
        node_text.append(emoji)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="text",
        text=node_text,
        textposition="middle center",
        hovertext=hover_texts,
        hoverinfo="text",
        textfont=dict(size=32),  # BIG emojis
        showlegend=False
    )

    # Emoji-only legend
    legend_items = []
    for col in columns:
        legend_items.append(go.Scatter(
            x=[None], y=[None],
            mode='text',
            text=[f"{category_emojis[col]} {col}"],
            textfont=dict(size=16),
            showlegend=True,
            name=f"{category_emojis[col]} {col}"
        ))

    fig = go.Figure(data=[node_trace] + legend_items)
    fig.update_layout(
        annotations=annotations,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=60),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )

    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)


