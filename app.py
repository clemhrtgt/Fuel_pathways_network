import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Load and clean data
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.str.strip().str.replace('\n', ' ') if col.dtype == "object" else col)

# Column order
columns = [
    "Feedstock Sources",
    "Energy  used  in  the process",
    "Process Type",
    "Fuel type"
]

# Emoji map
category_emojis = {
    "Feedstock Sources": "🌿",
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

    dcc.Graph(id="network-graph", style={"height": "1000px", "width": "100%"})
])

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

        x_map = {
            "Feedstock Sources": 1,
            "Energy  used  in  the process": 1,  # Même x que Feedstock
            "Process Type": 3,
            "Fuel type": 5
        }

        feedstock_nodes = sorted(node_groups["Feedstock Sources"])
        energy_nodes = sorted(node_groups["Energy  used  in  the process"])

        combined_nodes = []
        max_len = max(len(feedstock_nodes), len(energy_nodes))
        for i in range(max_len):
            if i < len(feedstock_nodes):
                combined_nodes.append((feedstock_nodes[i], "Feedstock Sources"))
            if i < len(energy_nodes):
                combined_nodes.append((energy_nodes[i], "Energy  used  in  the process"))

        y_spacing = 80

        for j, (node, cat) in enumerate(combined_nodes):
            x = x_map[cat] * 100
            y = -j * y_spacing
            pos[node] = (x, y)

        for col in ["Process Type", "Fuel type"]:
            nodes = sorted(node_groups[col])
            for j, node in enumerate(nodes):
                x = x_map[col] * 100
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
    for start, end in subG.edges():
        if start in pos and end in pos:
            x0, y0 = pos[start]
            x1, y1 = pos[end]
            annotations.append(create_arrow(x0, y0, x1, y1))

    emoji_x = []
    emoji_y = []
    emoji_text = []
    emoji_hovertext = []

    label_x = []
    label_y = []
    label_text = []

    for node in subG.nodes():
        x, y = pos[node]

        emoji = "❓"
        for col in columns:
            if node in df[col].values:
                emoji = category_emojis[col]
                break

        emoji_x.append(x)
        emoji_y.append(y)
        emoji_text.append(emoji)
        emoji_hovertext.append(node)

        label_x.append(x)
        label_y.append(y + 30)
        label_text.append(node)

    emoji_trace = go.Scatter(
        x=emoji_x,
        y=emoji_y,
        mode="text",
        text=emoji_text,
        textfont=dict(size=96),
        hoverinfo="text",
        hovertext=emoji_hovertext,
        showlegend=False
    )

    label_trace = go.Scatter(
        x=label_x,
        y=label_y,
        mode="text",
        text=label_text,
        textfont=dict(size=18),
        textposition="top center",
        hoverinfo="skip",
        showlegend=False
    )

    legend_items = []
    for col in columns:
        legend_items.append(go.Scatter(
            x=[None],
            y=[None],
            mode='text',
            text=[f"{category_emojis[col]} {col}"],
            textfont=dict(size=16),
            showlegend=True,
            name=f"{category_emojis[col]} {col}"
        ))

    fig = go.Figure(data=[emoji_trace, label_trace] + legend_items)
    fig.update_layout(
        annotations=annotations,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=60),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)


