import numpy as np
import subprocess
import os
import networkx as nx
import torch  # this is needed to fix the buggy import of open3d in the next line.
from deconstruct.utils.open3d_utils.general_utils import get_absolute_path_to_repo

PATH_KAMIS_EXEC = os.path.join(get_absolute_path_to_repo(),
                               "deconstruct/proposal_selection/kamis_solvers")


class MWISSolver:
    default_parameter_dict = {"reduction_style": "dense", "time_limit": 20.}

    def __init__(self, method="weighted_local_search", parameter_dict=None, path_tmp_graph_file="default",
                 path_tmp_solution_file="default", verbose=True):
        assert method in ["weighted_branch_reduce", "weighted_local_search"], f"method {method} not implemented"
        self.verbose = verbose
        self.method = method
        self.parameter_dict = self.default_parameter_dict.copy()
        self.unique_identifier = np.random.randint(0, 1e+9)
        if parameter_dict is not None:
            self.parameter_dict.update(parameter_dict)
        self.path_tmp_graph_file = path_tmp_graph_file
        if self.path_tmp_graph_file == "default":
            self.path_tmp_graph_file = os.path.join(PATH_KAMIS_EXEC, f"kamis_in_tmp_{self.unique_identifier}.txt")
        self.path_tmp_solution_file = path_tmp_solution_file
        if self.path_tmp_solution_file == "default":
            self.path_tmp_solution_file = os.path.join(PATH_KAMIS_EXEC, f"kamis_out_tmp_{self.unique_identifier}.txt")

    @staticmethod
    def rescale_node_weights(G, sum_limit=1e+9):
        # sum limit should be not more than 1e+9 to avoid overflow in KaMIS
        node_weights = np.asarray([G.nodes[node]["weight"] for node in range(G.number_of_nodes())])
        node_weights = node_weights / np.sum(node_weights) * sum_limit

        return node_weights.astype(int)

    def graph_to_KaMIS_file(self, G):
        # use convention as specified in the user manual
        # (https://github.com/KarlsruheMIS/KaMIS/blob/master/docs/user_guide.pdf)
        node_weights_scaled = self.rescale_node_weights(G)

        lines = [f"{G.number_of_nodes()} {G.number_of_edges()} 10\n"]  # 10 indicates that graph has node weights
        for i in range(G.number_of_nodes()):
            # the plus 1 is because labelling starts from 1 in KaMIS
            lines.append(f"{node_weights_scaled[i]} {''.join([str(n + 1) + ' ' for n in G.neighbors(i)])}\n")

        # print(lines)
        with open(self.path_tmp_graph_file, "w") as f:
            f.writelines(lines)

    def solution_from_KaMIS_file(self):
        with open(self.path_tmp_solution_file, "r") as f:
            indicator_str = f.read()
        indicator_arr = np.asarray([int(x) for x in indicator_str.split("\n")[:-1]])

        return indicator_arr

    def __call__(self, G):
        node_weights = np.asarray([G.nodes[node]["weight"] for node in range(G.number_of_nodes())])
        assert (node_weights > 0).all(), "node weights must be truly positive."
        self.graph_to_KaMIS_file(G)

        # call solver:
        exec_list = [f"./{self.method}", self.path_tmp_graph_file, "--output", self.path_tmp_solution_file]
        for parameter in self.parameter_dict:
            exec_list += [f"--{parameter}", str(self.parameter_dict[parameter])]

        subprocess.run(exec_list, cwd=PATH_KAMIS_EXEC, capture_output=np.invert(self.verbose))

        indicator_arr = self.solution_from_KaMIS_file()
        accepted_nodes = np.arange(G.number_of_nodes())[(indicator_arr).astype(bool)]
        weight_sum = indicator_arr.dot(node_weights)

        # remove tmp files:
        os.remove(self.path_tmp_graph_file)
        os.remove(self.path_tmp_solution_file)

        return weight_sum, accepted_nodes


if __name__ == '__main__':
    # generate a graph for MWIS problem:

    G = nx.gnp_random_graph(n=100, p=0.1, seed=None, directed=False)

    # generate random node weights:
    node_weights = np.random.random(G.number_of_nodes())
    nx.set_node_attributes(G, {i: w for i, w in enumerate(node_weights)}, name="weight")
    # KaMIS_MWIS_solver(G, method="weighted_local_search",
    #                   parameter_dict={"reduction_style": "dense", "time_limit": 20.},
    #                   verbose=True)
    mwis_solver = MWISSolver(parameter_dict={"reduction_style": "dense", "time_limit": 20.})
    weight_sum, selected_indices = mwis_solver(G)
