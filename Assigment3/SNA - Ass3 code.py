import networkx as nx


def get_name():
    return 'Ayelet Hashahar Cohen'

def get_id():
    return '206533895'


# ----------------- Question 1 -----------------

def centrality_measures(network, node, iterations=100):
    """
    :param network: The network to run the analysis over
    :param node: The node name (represented as integer) to retrieve the centrality measures for. Although in some
     networks, node names can take a string - we deal only with integer values.
    :param iterations: To be used by the page-rank and the authority score algorithms
    :return: dictionary with the following keys: dc, cs, nbc, pr, auth
    """
    dc = nx.degree_centrality(network).get(node, 0.0)
    cs = nx.closeness_centrality(network).get(node, 0.0)
    bc = nx.betweenness_centrality(network, normalized=True).get(node, 0.0)
    pr = nx.pagerank(network, max_iter=iterations, alpha=0.85).get(node, 0.0)
    _, authorities = nx.hits(network, max_iter=iterations)
    auth = authorities.get(node, 0.0)

    return {
        'dc': dc,
        'cs': cs,
        'nbc': bc,
        'pr': pr,
        'auth': auth
    }

def single_step_voucher(network):
    """
    :param network: The network to run the analysis over.
    :return: the best node name to send the voucher to.
    """
    degree_centrality = nx.degree_centrality(network)
    best_node = max(degree_centrality, key=degree_centrality.get)
    return best_node


def multiple_steps_voucher(network):
    """
    :param network: The network to run the analysis over.
    :return: the best node name to send the voucher to.
    """
    closeness_centrality = nx.closeness_centrality(network)
    best_node = max(closeness_centrality, key=closeness_centrality.get)
    return best_node


def calculate_total_benefit(network, start_node, reduction_per_step=0.06, max_steps=4):
    total_benefit = 0.0
    for node in network:
        if node == start_node:
            continue
        try:
            shortest_path_length = nx.shortest_path_length(network, source=start_node, target=node)
            if shortest_path_length <= max_steps:
                benefit = (1 - reduction_per_step) ** shortest_path_length
                total_benefit += benefit
        except nx.NetworkXNoPath:
            continue
    return total_benefit


def multiple_steps_diminished_voucher(network):
    """
    :param network: The network to run the analysis over.
    :return: the best node name to send the voucher to.
    """
    best_node = None
    max_benefit = -1
    for node in network:
        benefit = calculate_total_benefit(network, node)
        if benefit > max_benefit:
            max_benefit = benefit
            best_node = node
    return best_node


def find_most_valuable(network):
    betweenness_centrality = nx.betweenness_centrality(network, normalized=True)
    most_valuable_node = max(betweenness_centrality, key=betweenness_centrality.get)

    return most_valuable_node




# Example usage:
# Create a sample graph
# Load the network from a GML file
# network = nx.read_gml(r'C:\Users\ahash\OneDrive - post.bgu.ac.il\Data engineering - bcs\Semester 6\Social network analysis\Assigments\Ass3\friendships.gml.txt')

# # Calculate centrality measures for node 0
# result_1 = centrality_measures(network, 1)
# print("centrality measures of node 1: ", result_1)
# result_50 = centrality_measures(network, 50)
# print("centrality measures of node 50: ", result_50)
# result_100 = centrality_measures(network, 100)
# print("centrality measures of node 100: ", result_100)

# Calculate the best node to send the voucher to
# best_node = single_step_voucher(network)
# print("best node to send the voucher to: ", best_node)

# best_node = multiple_steps_voucher(network)
# print("best node to send the voucher to: ", best_node)

# best_node = multiple_steps_diminished_voucher(network)
# print("best node to send the voucher to: ", best_node)

# best_node = find_most_valuable(network)
# print("best node to send the voucher to: ", best_node)