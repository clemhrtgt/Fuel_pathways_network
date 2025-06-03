import dash
from dash import dcc, html, Input, Output
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

df = pd.read_csv("fuel_pathways.csv")

columns = [
    "Feedstock Sources",
    "Energy  used  in  the process",
    "Process Type",
    "Fuel type"
]

category_emojis = {
    "Feedstock Sources": "🌱",
    "Energy  used  in  the process": "⚡",
    "Process Type": "🏭",
    "Fuel type": "⛽"
}

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id="pathway-filter",
        options=[{"label": code, "value": code} for code in df["Fuel Pathway Code"].unique()],
        multi=True,
        placeholder="Select Fuel Pathway Code(s)..."
    ),
    dcc.Graph(id="network-graph")
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
            subG.add_edge(fs, pt, type="feedstock")
        if pd.notna(en) and pd.notna(pt):
            subG.add_edge(en, pt, type="energy")
        if pd.notna(pt) and pd.notna(ft):
            subG.add_edge(pt, ft, type="process_to_fuel")

    def get_custom_layout():
        pos = {}
        node_groups = {col: set() for col in columns}
        for _, row in filtered.iterrows():
            for col in columns:
                if pd.notna(row[col]):
                    node_groups[col].add(row[col])

        x_map = {
            "Feedstock Sources": 0,
            "Energy  used  in  the process": 2,
            "Process Type": 4,
            "Fuel type": 6
        }

        y_spacing = 80
        for col in columns:
            nodes = sorted(node_groups[col])
            for j, node in enumerate(nodes):
                x = x_map[col] * 100
                y = -j * y_spacing
                pos[node] = (x, y)
        return pos

    pos = get_custom_layout()

    offset = 15  # offset to separate feedstock and energy arrows

    annotations = []
    for u, v, data in subG.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        if data.get("type") == "feedstock":
            # Shift feedstock arrow slightly up to avoid overlap
            x0_adj = x0
            y0_adj = y0 + offset
            x1_adj = x1 - offset
            y1_adj = y1 + offset
        elif data.get("type") == "energy":
            # Shift energy arrow slightly down to avoid overlap
            x0_adj = x0
            y0_adj = y0 - offset
            x1_adj = x1 - offset
            y1_adj = y1 - offset
        else:
            x0_adj, y0_adj, x1_adj, y1_adj = x0, y0, x1, y1

        annotations.append(dict(
            ax=x0_adj, ay=y0_adj,
            x=x1_adj, y=y1_adj,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="black",
            opacity=1
        ))

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
        label_y.append(y - 20)  # Label just below the node (closer than before)
        label_text.append(node)

    emoji_trace = go.Scatter(
        x=emoji_x,
        y=emoji_y,
        mode="text",
        text=emoji_text,
        textfont=dict(size=32),
        hoverinfo="text",
        hovertext=emoji_hovertext,
        showlegend=False
    )

    label_trace = go.Scatter(
        x=label_x,
        y=label_y,
        mode="text",
        text=label_text,
        textfont=dict(size=12),
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
        yaxis=dict() 
    )

    return fig

if __name__ == "__main__":
    app.run_server(debug=True) 
    
