import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.widgets import CheckButtons


def create_network_from_table_data(table_df: pd.DataFrame, round_num: int, network_graph_ax: plt.Axes):
    graph = nx.DiGraph()
    round_df = table_df.loc[table_df['round'] == round_num]
    round_df = round_df.sort_values(by='team_abbr', ascending=False)

    round_df[['team_abbr', 'team_color']].apply(lambda row: graph.add_node(row['team_abbr'], color=row['team_color']),
                                                axis=1)

    edges_df = round_df.loc[round_df['match_gd_full'] < 0]

    edges_df[['team_abbr', 'opp_team_abbr', 'match_gd_full']].apply(lambda row:
                                                                    graph.add_edge(row['team_abbr'],
                                                                                   row['opp_team_abbr'],
                                                                                   weight=-row['match_gd_full']),
                                                                    axis=1)

    weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    colors = [graph.nodes[u]['color'] for u in graph.nodes()]

    nx.draw_circular(graph,
                     ax=network_graph_ax,
                     with_labels=True,
                     font_weight='bold',
                     width=weights,
                     node_color=colors,
                     font_size=8,
                     node_size=500,
                     font_color='white'
                     )
