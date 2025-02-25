import networkx as nx
import random
import numpy as np
from scipy.stats import binom_test
# import pickle
import powerlaw
# import matplotlib.pyplot as plt
# from scipy.stats import probplot

# with open('rand_nets.p', 'rb') as f:
#     rand_nets_networks = pickle.load(f)

# with open('scalefree_nets.p', 'rb') as f:
#     scalefree_nets_networks = pickle.load(f)

# with open('mixed_nets.p', 'rb') as f:
#     mixed_nets_networks = pickle.load(f)


def get_name():
    return 'Ayelet Hashahar Cohen'

def get_id():
    return '206533895'

# ------------------------------- Q1 -------------------------------
def random_networks_generator(n, p, num_networks = 1, directed = False, seed = 206533895):
    """
        Generates a list of random graphs based on the G(n, p) Erdős-Rényi model.

        Parameters:
        n (int): Number of nodes in each graph. Must be a non-negative integer.
        p (float): Probability of creating an edge between two nodes in the graph.
                  This value should be between 0 and 1, inclusive.
        num_networks (int, optional): Number of random graphs to generate. Defaults to 1.
        directed (bool, optional): If True, generates directed graphs. Defaults to False.
        seed (int, optional): Seed for the random number generator. This is used to ensure
                              reproducibility. Defaults to 206533895.

        Returns:
        list: A list containing `num_networks` networkx graph objects, each representing
              a random graph generated according to the G(n, p) model. If `directed` is True,
              the graphs will be directed; otherwise, they will be undirected.

        Notes:
        - Each graph is initialized with the same seed, but the seed is incremented after
          generating each graph to ensure each graph is different even if `num_networks` > 1.
        - The function uses the networkx library's gnp_random_graph function to generate graphs.
        """
    lst_networks:list = []
    for i in range(num_networks):
        if directed:
            G = nx.gnp_random_graph(n, p, seed=seed, directed=True)
        else:
            G = nx.gnp_random_graph(n, p, seed=seed, directed=False)
        lst_networks.append(G)
        # Update seed for each network to ensure different networks
        seed += 1
    return lst_networks

def network_stats(network):
    """
        Calculates and returns various statistical metrics about the given network graph.

        Parameters:
        network (networkx.Graph or networkx.DiGraph): The network graph object for which statistics are to be computed.

        Returns:
        dict: A dictionary containing statistical metrics of the network including average degree, standard deviation
              of degrees, minimum degree, maximum degree, average shortest path length (spl), and diameter. If the graph is
              not connected (or strongly connected in case of directed graphs), spl and diameter are set to infinity.

    """
    degrees = np.array([degree for _, degree in network.degree()])
    dict_statistics = {
        # Average degrees distribution.
        'degrees_avg': np.mean(degrees),
        # Standard deviation degrees distribution.
        'degrees_std': np.std(degrees),
        # Degrees distribution minimum value.
        'degrees_min': np.min(degrees),
        # Degrees distribution maximum value.
        'degrees_max': np.max(degrees)
    }

    # Calculate average shortest path length and diameter if the graph is strongly connected or connected
    if network.is_directed():
        if nx.is_strongly_connected(network):
            dict_statistics['spl'] = nx.average_shortest_path_length(network)
            dict_statistics['diameter'] = nx.diameter(network)
        else:
            dict_statistics['spl'] = float('inf')  # or some other placeholder
            dict_statistics['diameter'] = float('inf')  # or some other placeholder
    else:
        if nx.is_connected(network):
            dict_statistics['spl'] = nx.average_shortest_path_length(network)
            dict_statistics['diameter'] = nx.diameter(network)
        else:
            dict_statistics['spl'] = float('inf')  # or some other placeholder
            dict_statistics['diameter'] = float('inf')  # or some other placeholder

    return dict_statistics

def networks_avg_stats(networks):
    """
        Calculates average network statistics across a list of network graphs.

        Parameters:
        networks (list of networkx.Graph or networkx.DiGraph): A list of network graph objects.

        Returns:
        dict: A dictionary containing the average of statistical metrics across all given networks, including
              average degree, standard deviation of degrees, minimum and maximum degrees, average shortest path length,
              and diameter. If any network is not fully connected, 'spl' and 'diameter' might be infinity.
    """
    list_avg_stats = []
    for network in networks:
        list_avg_stats.append(network_stats(network))

        # Initialize dictionary to store average statistics
    avg_stats = {
        'degrees_avg': 0,
        'degrees_std': 0,
        'degrees_min': 0,
        'degrees_max': 0,
        'spl': 0,
        'diameter': 0
    }

    # Calculate the sum of each statistic across all networks
    for stats in list_avg_stats:
        for key in avg_stats:
            avg_stats[key] += stats[key]

    # Calculate average of each statistic
    num_networks = len(networks)
    for key in avg_stats:
        avg_stats[key] /= num_networks

    return avg_stats

# ------------------------------- Q2 -------------------------------
def rand_net_hypothesis_testing(network, theoretical_p, alpha = 0.05):
    """
        Conducts a hypothesis test using a binomial test to determine if the number of edges in a network
        significantly differs from a binomially distributed number of edges expected by chance, given a
        theoretical probability of edge formation.

        Parameters:
        network (networkx.Graph or networkx.DiGraph): The network graph object being tested.
        theoretical_p (float): The theoretical probability of an edge forming between any two nodes in a completely random network.
        alpha (float, optional): Significance level used to decide whether to reject the null hypothesis (default is 0.05).

        Returns:
        tuple: A tuple containing the p-value of the test and a string indicating whether the null hypothesis
               is "accept"ed or "reject"ed. The null hypothesis in this context is that the network's edge
               formation does not significantly differ from the theoretical model.
    """
    n = len(network.nodes())
    actual_edges = len(network.edges())

    # Using a binomial test to test the number of edges
    p_value = binom_test(actual_edges, int(n * (n - 1) / 2), theoretical_p)

    # Determine if we reject or accept the null hypothesis
    if p_value < alpha:
        return (p_value, "reject")
    else:
        return (p_value, "accept")

def most_probable_p(graph):
    """
        Identifies the most plausible probability of edge formation (p) that might have generated the given graph,
        from a set of predefined probability values, using hypothesis testing.

        Parameters:
        graph (networkx.Graph or networkx.DiGraph): The network graph object for which to find the most plausible p-value.

        Returns:
        float: The probability value (from the tested set) that has the smallest p-value from hypothesis testing which
               is accepted (i.e., does not significantly differ from the graph), or -1 if no such value is accepted.
    """
    dict_prob = {}
    for prob in [0.01, 0.1, 0.3, 0.6]:
        dict_prob[prob] = rand_net_hypothesis_testing(graph, prob)

    max_p_value = -1
    for key, item in dict_prob.items():
        if item[1] == "accept" and item[0] > max_p_value:
            max_p_value = item[0]
            max_prob = key

    if max_p_value == -1:
        return -1
    else:
        return max_prob

# ------------------------------- Q3 -------------------------------

def find_opt_gamma(network, treat_as_social_network = True):
    """
        Estimates the optimal 𝛾 parameter for a given network using the powerlaw package.

        :param network: A networkX graph object representing the network.
        :param treat_as_social_network: A boolean flag to indicate if the network is a social network.
        :return: A float representing the optimal 𝛾 parameter.
        """
    # Extract degrees from the network
    degrees = [d for n, d in network.degree()]

    # Fit the degrees to a powerlaw distribution
    fit = powerlaw.Fit(degrees, discrete=treat_as_social_network, verbose=False)

    # Return the estimated power-law exponent
    gamma = fit.power_law.alpha
    return gamma

# ------------------------------- Q4 -------------------------------

def netwrok_classifier(network):
    """
    Classify a network as random or scale-free.

    :param: network: A networkx Graph object representing the network.
    :return: 1 if the network is classified as random, 2 if it is classified as scale-free.
    """
    gamma = find_opt_gamma(network)
    if gamma >= 3:
        return 1
    elif gamma > 2 and gamma < 3:
        return 2
    return 2


# for index, net in enumerate(mixed_nets_networks):
#     if netwrok_classifier(net) == 1:
#         print(f"network {index+1} is a random network")
#     else:
#         print(f"network {index+1} is a scale-free network")

# lst = []
# for index, network in enumerate(scalefree_nets_networks):
#     print(f"network {index+1} has gamma {find_opt_gamma(network)}")

# print(network_stats(scalefree_nets_networks[0]))
# print(network_stats(rand_nets_networks[6]))
# list_network = random_networks_generator(100,0.1)
# print(most_probable_p(list_network[0]))

# list_network = random_networks_generator(100,0.1, 20)
# print("Type a:\n num_networks = 20, n = 100, p = 0.1")
# print(networks_avg_stats(list_network))
#
# list_network = random_networks_generator(100,0.6, 20)
# print("Type b:\n num_networks = 20, n = 100, p = 0.6")
# print(networks_avg_stats(list_network))
#
# list_network = random_networks_generator(1000,0.1, 10)
# print("Type c:\n num_networks = 10, n = 1000, p = 0.1")
# print(networks_avg_stats(list_network))
#
# list_network = random_networks_generator(1000,0.6, 10)
# print("Type d:\n num_networks = 10, n = 1000, p = 0.6")
# print(networks_avg_stats(list_network))
#
# print(len(rand_nets_networks))
# lst = []
# for network in rand_nets_networks:
#     lst.append(most_probable_p(network))
# print(lst)
# my_network = rand_nets_networks[0]
# print(f"for p: {0.65} -> {rand_net_hypothesis_testing(my_network, (0.65))}")
# print(f"for p: {0.55} -> {rand_net_hypothesis_testing(my_network, (0.55))}")
# print(f"for p: {0.7} -> {rand_net_hypothesis_testing(my_network, (0.7))}")
# print(f"for p: {0.5} -> {rand_net_hypothesis_testing(my_network, (0.5))}")
#
# list_network = random_networks_generator(1000,0.1)
# print(most_probable_p(list_network[0]))
# print(f"for p: {0.11} -> {rand_net_hypothesis_testing(list_network[0], (0.11))}")
# print(f"for p: {0.09} -> {rand_net_hypothesis_testing(list_network[0], (0.09))}")
# print(f"for p: {0.2} -> {rand_net_hypothesis_testing(list_network[0], (0.2))}")
# print(f"for p: {0.01} -> {rand_net_hypothesis_testing(list_network[0], (0.01))}")


# Load your network list first
# my_network = list_networks[0]  # choose the first network
#
# # Generate a random network with the same number of nodes and edges
# random_network = nx.gnm_random_graph(my_network.number_of_nodes(), my_network.number_of_edges())
#
# # Calculate the degree distribution for the given network and the random network
# given_degrees = list(dict(my_network.degree()).values())
# random_degrees = list(dict(random_network.degree()).values())
#
# # Plot the QQ plot
# fig, ax = plt.subplots()
# stats.probplot(given_degrees, dist="norm", plot=ax)  # Plot for the given network
# stats.probplot(random_degrees, dist="norm", plot=ax)  # Plot for the random network
#
# ax.legend(["My network", "Random network"])
# ax.set_title("QQ plot")
# plt.show()