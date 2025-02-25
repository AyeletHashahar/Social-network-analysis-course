import os
import json
import csv
from datetime import datetime
from collections import defaultdict, Counter
import networkx as nx

def get_name():
    return 'Ayelet Hashahar Cohen'

def get_id():
    return '206533895'

# ----------------- Question 1 -----------------

def calculate_modularity(network, communities):
    """
    Calculate the modularity of the given partition of communities in the network.

    Parameters:
    network (NetworkX graph): The input network.
    communities (list of lists): A list of communities, each community is a list of nodes.

    Returns:
    float: The modularity value of the partition.
    """
    L = network.number_of_edges()  # Total number of edges in the network
    total_modularity = 0

    for community in communities:
        subgraph = network.subgraph(community)  # Subgraph for the current community
        L_c = subgraph.number_of_edges()  # Number of edges within the community
        k_c = sum(dict(network.degree(community)).values())  # Sum of degrees of nodes in the community
        total_modularity += ((L_c / L) - (k_c / (2 * L)) ** 2)  # Calculate modularity for the community

    return total_modularity


def community_detector(algorithm_name, network, most_valuable_edge = None):
    # Initialize variables
    partition = []
    modularity_value = -1
    current_modularity = -1

    # Girvan-Newman algorithm
    if algorithm_name == 'girvin_newman':
        # Apply Girvan-Newman algorithm
        comp = nx.community.girvan_newman(network, most_valuable_edge)
        max_modularity = -1
        best_partition = None
        # Iterate through the resulting communities
        for communities in comp:
            current_partition = [list(community) for community in communities]
            # Calculate modularity for the current partition
            current_modularity = nx.community.modularity(network, current_partition)
            # Update the best partition if current modularity is higher
            if current_modularity > max_modularity:
                max_modularity = current_modularity
                best_partition = current_partition
        partition = best_partition
        modularity_value = max_modularity

    # Louvain algorithm
    elif algorithm_name == 'louvain':
        # Apply Louvain algorithm
        community = list(nx.community.louvain_communities(network))
        partition = [list(c) for c in community]
        # Calculate modularity for the partition
        modularity_value = nx.community.modularity(network, community)

    # Clique Percolation algorithm
    elif algorithm_name == 'clique_percolation':
        # Determine the size of the largest clique
        largest_clique = nx.algorithms.approximation.clique.large_clique_size(network)
        # Iterate through possible clique sizes
        for k in range(3, largest_clique + 1):
            # Apply k-clique percolation
            community = nx.community.k_clique_communities(network, k)
            optional_partition = [list(c) for c in community]
            cal_all_communities = optional_partition
            # Calculate modularity for the current partition
            modularity_val = calculate_modularity(network, cal_all_communities)
            # Update the best partition if current modularity is higher
            if current_modularity <= modularity_val:
                current_modularity = modularity_val
                partition = optional_partition
        modularity_value = current_modularity


    else:
        raise ValueError('Unknown algorithm name')


    return {'num_partitions': len(partition), 'modularity': modularity_value, 'partition': partition}


def edge_selector_optimizer(G):
    betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    return max(betweenness, key=betweenness.get)

# -----------------------------------------------
# ----------------- Question 2 ------------------


def read_central_players(files_path):
    central_players = set()
    central_players_file = os.path.join(files_path, 'central_political_players.csv')
    with open(central_players_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            central_players.add(row[0])
    return central_players

def extract_date_from_filename(filename):
    try:
        date_str = filename.split('.')[-2]  # Extract the date part before the last extension
        return datetime.strptime(date_str, '%Y-%m-%d')
    except Exception as e:
        print(f"Error extracting date from filename {filename}: {e}")
        return None


def construct_heb_edges(files_path, start_date='2019-03-15', end_date='2019-04-15', non_parliamentarians_nodes=0):
    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    central_players = read_central_players(files_path)

    edges = defaultdict(int)
    non_parliamentarian_edges = defaultdict(int)
    retweeted_counter = Counter()

    # Process each txt file in the files_path
    for filename in os.listdir(files_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(files_path, filename)
            tweet_date = extract_date_from_filename(filename)  # Extract date from filename
            if start_date <= tweet_date <= end_date:
                with open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            tweet = json.loads(line)
                            if 'retweeted_status' in tweet:
                                retweeted_user_id = tweet['retweeted_status']['user']['id_str']
                                retweeter_user_id = tweet['user']['id_str']
                                if retweeted_user_id in central_players and retweeter_user_id in central_players:
                                    edge = (retweeter_user_id, retweeted_user_id)
                                    edges[edge] += 1

                                # Add non-parliamentarian nodes to the edges dictionary
                                # and count the number of retweets by central players for the parliamentarian nodes
                                if non_parliamentarians_nodes > 0:
                                    if retweeted_user_id not in central_players or retweeter_user_id not in central_players:
                                        if retweeter_user_id in central_players:
                                            retweeted_counter[retweeted_user_id] += 1
                                            edge = (retweeter_user_id, retweeted_user_id)
                                            non_parliamentarian_edges[edge] += 1
                        except Exception as e:
                            print(f"Error processing line: {e}")

    try:
        if non_parliamentarians_nodes > 0:
            nodes_to_add = [item[0] for item in retweeted_counter.most_common(non_parliamentarians_nodes)]
            central_players.update(nodes_to_add)

            # Sort the non-parliamentarian edges by retweet count and add the top nodes to the edges dictionary
            for key, val in non_parliamentarian_edges.items():
                if key[0] in central_players and key[1] in central_players:
                    edges[key] = val
    except Exception as e:
        print(f"Error adding non-parliamentarian nodes: {e}")

    return edges


def construct_heb_network(edge_dict):
    graph_tweet = nx.DiGraph()
    for edge, weight in edge_dict.items():
        graph_tweet.add_edge(edge[1], edge[0], weight=weight)
    return graph_tweet


# import matplotlib.pyplot as plt
# import networkx as nx
#
#
# def read_names_mapping(files_path):
#     id_to_name = {}
#     names_file = os.path.join(files_path, 'central_political_players.csv')
#     with open(names_file, 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # Skip header
#         for row in reader:
#             id_to_name[row[0]] = row[1]
#     return id_to_name
#
#
# def plot_communities_with_names(graph, communities, title, modularity, filename, id_to_name):
#     plt.figure(figsize=(12, 12))
#     pos = nx.spring_layout(graph, seed=42)
#     colors = plt.get_cmap('tab20').colors  # Select a color map with distinct colors
#
#     for i, community in enumerate(communities):
#         color = colors[i % len(colors)]
#         nx.draw_networkx_nodes(graph, pos, nodelist=community, node_color=[color], node_size=50,
#                                label=f'Community {i + 1}')
#
#     nx.draw_networkx_edges(graph, pos, alpha=0.3)
#
#     # Map IDs to names for labels
#     labels = {node: id_to_name.get(node, node) for node in graph.nodes()}
#     nx.draw_networkx_labels(graph, pos, labels, font_size=8)
#
#     plt.title(f"{title}\nModularity: {modularity}")
#     plt.legend()
#     plt.savefig(filename)
#     plt.show()
#
# def plot_network(graph, title):
#     plt.figure(figsize=(10, 10))
#     pos = nx.spring_layout(graph, seed=42)
#     nx.draw_networkx_nodes(graph, pos, node_size=50)
#     nx.draw_networkx_edges(graph, pos, alpha=0.3)
#     plt.title(title)
#     plt.show()
#
# def plot_modularity_vs_non_parliamentarians(data):
#     plt.figure(figsize=(10, 6))
#     plt.plot(data['non_parliamentarians'], data['modularity'], marker='o')
#     plt.xlabel('Non Parliamentarians Nodes')
#     plt.ylabel('Modularity')
#     plt.title('Non Parliamentarians Nodes vs Modularity')
#     plt.grid(True)
#     plt.show()
#
# if __name__ == '__main__':
    # question 1
    # G1 = nx.les_miserables_graph()
    # result_girvin_newman = community_detector('girvin_newman', G1)
    # result_girvin_newman_with_optimizer = community_detector('girvin_newman', G1, edge_selector_optimizer)
    # result_louvain = community_detector('louvain', G1)
    # result_clique_percolation = community_detector('clique_percolation', G1)
    # print('Girvan-Newman: ', result_girvin_newman)
    # print('Girvan-Newman with edge_selector_optimizer: ', result_girvin_newman_with_optimizer)
    # print('Louvain: ', result_louvain)
    # print('Clique Percolation: ', result_clique_percolation)

    # question 2
    # files_path = r'twitter_files'
    # edges = construct_heb_edges(files_path, non_parliamentarians_nodes=10)
    # heb_network = construct_heb_network(edges)
    #
    # result_heb_gn = community_detector('girvin_newman', heb_network, edge_selector_optimizer)
    #
    # print(f"Number of partitions: {result_heb_gn['num_partitions']}")
    # print(f"Modularity: {result_heb_gn['modularity']}")
    # for i, community in enumerate(result_heb_gn['partition']):
    #     print(f"Community {i + 1}: {community}")
    #
    # id_to_name = read_names_mapping(files_path)  # Load the ID to name mapping
    #
    # plot_communities_with_names(heb_network, result_heb_gn['partition'],
    #                             'Girvin-Newman Algorithm on Hebrew Twitter Data', result_heb_gn['modularity'],
    #                             'girvin_newman_hebrew_twitter.png', id_to_name)
    #
    # non_parliamentarians_nodes = 10
    # edges = construct_heb_edges(files_path, non_parliamentarians_nodes=10)  # הוספת קודקודים לא מרכזיים
    # heb_network = construct_heb_network(edges)
    #
    # result_heb_gn = community_detector('girvin_newman', heb_network, edge_selector_optimizer)
    #
    # print(f"Number of partitions: {result_heb_gn['num_partitions']}")
    # print(f"Modularity: {result_heb_gn['modularity']}")
    # for i, community in enumerate(result_heb_gn['partition']):
    #     print(f"Community {i + 1}: {community}")
    #
    # plot_communities(heb_network, result_heb_gn['partition'], 'Girvin-Newman Algorithm on Hebrew Twitter Network',
    #                  result_heb_gn['modularity'], 'girvin_newman_hebrew_twitter.png')

    # plot_network(heb_network, 'Hebrew Twitter Network')

    # Data for the modularity plot
    # modularity_data = {
    #     'non_parliamentarians': [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0],
    #     'modularity': [0.798, 0.75, 0.65, 0.6, 0.55, 0.5, 0.47, 0.46, 0.465]
    # }
    # plot_modularity_vs_non_parliamentarians(modularity_data)