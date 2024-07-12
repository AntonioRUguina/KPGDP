from Utils.objects import (Candidate)
import numpy as np
import random


class Solution:
    def __init__(self, param_dict, max_of, active_ls3):
        self.instance = param_dict["inst"]
        self.max_of = max_of
        self.sorted_distances = self.instance.sorted_distances.copy()
        self.distance = self.instance.distance
        self.selected_list = []
        self.selected_dict = {k: [] for k in range(0, self.instance.k)}
        self.n_selected = self.instance.k*[0]
        self.v_min1 = -1
        self.v_min2 = -1
        self.of = self.sorted_distances[0].distance * 10
        self.dict_disp_group = {k: [] for k in range(0, self.instance.k)}
        self.groups = self.instance.k
        self.historial = []
        self.p = self.instance.p
        self.group_last_selection = None
        self.coef_bound = 0.7
        self.end_iteration = False
        self.integers_list = []
        self.active_ls3 = active_ls3

    def generate_random_ordered_integers(self):
        # Step 1: Generate a list of the first n integers
        integers_list = list(range(self.instance.n))

        # Step 2: Shuffle the list randomly
        random.shuffle(integers_list)
        self.integers_list = integers_list

    def run_algorithm(self, beta_0, beta_1):
        self.generate_random_ordered_integers()
        self.construct_solution(beta_0, beta_1)
        max_ls = 100
        self.historial.append(self.of)
        if not self.end_iteration:
            self.run_exchage_LS(max_ls, self.active_ls3)

        return self.of

    def run_algorithm_chained(self, beta_0, beta_1):
        self.generate_random_ordered_integers()
        self.coef_bound = 0
        list_of = []
        for k in range(self.groups):
            self.groups = 1
            self.of = 999999
            #self.selected_dict = {0: []}
            #self.dict_disp_group = {0: []}
            self.construct_solution_chained(beta_0, beta_1, k)
            list_of.append(self.of)
        max_ls = 100
        #if not self.end_iteration:
            #self.run_exchage_LS(max_ls, k)


        return min(list_of)

    def save_dict_to_txt(self, filename, d_value, instance_name, model, time, seed):
        with open(filename, 'a') as file:
                file.write(f"{instance_name} {seed} {model} {time}: {d_value} \n")

    def update_of(self, v_min1, v_min2, of, group, recalculate= False):
        if recalculate:
            new_of = 999999999
            for v1 in self.selected_dict[group]:
                for v2 in self.selected_dict[group]:
                    if v1 > v2:
                        dist = self.instance.distance[v1][v2]
                        if new_of > dist:
                            new_of = dist
                            new_v1_min = v1
                            new_v2_min = v2
            self.dict_disp_group[group] = [(new_v1_min, new_v2_min), new_of]
            self.of = min([self.dict_disp_group[k][1] for k in range(self.groups)])

        else:
            if len(self.dict_disp_group[group]) == 0:
                self.dict_disp_group[group] = [(v_min1, v_min2), of]

            elif self.dict_disp_group[group][1] > of:
                self.dict_disp_group[group] = [(v_min1, v_min2), of]

            if self.of > of:
                self.of = of
                self.v_min1 = v_min1
                self.v_min2 = v_min2


    def add(self, v, k):
        self.selected_dict[k].append(v)
        self.selected_list.append(v)
        self.n_selected[k] += 1
        if self.n_selected[k] == self.p - 1:
            self.group_last_selection = k

    def is_feasible(self):
        feasible = True
        for key, value in self.selected_dict.items():
            feasible = feasible and len(value) >= self.p
            if not feasible:
                break

        return feasible

    def find_first_with_group(self, cl, target_group):
        for index, obj in enumerate(cl):
            if obj.group == target_group:
                return index
        return None


    def construct_solution(self, beta_0, beta_1):

        for k in range(0, self.groups):
            stop = True
            while stop:
                index_selected = int((np.log(np.random.random()) / np.log(1 - beta_0))) % len(self.sorted_distances)
                edge = self.sorted_distances[index_selected]
                if (edge.v1 not in self.selected_list and edge.v2 not in self.selected_list):
                    self.add(edge.v1, k)
                    self.add(edge.v2, k)
                    self.update_of(edge.v1, edge.v2, edge.distance, k)
                    stop = False
                self.sorted_distances.pop(index_selected)

        cl = self.create_cl()

        # 0 dado que es el greedy
        while not self.is_feasible():
            if len(cl) < 1:
                self.of = 0
                self.end_iteration = True
                break
            if self.group_last_selection is not None:
                index_selected = self.find_first_with_group(cl, self.group_last_selection)
                self.group_last_selection = None
            else:
                index_selected = int((np.log(np.random.random()) / np.log(1 - beta_1))) % len(cl)

            if index_selected is None:
                self.of = 0
                self.end_iteration = True
                break

            else:
                c = cl.pop(index_selected)
                self.add(c.v, c.group)
                self.update_of(c.v, c.closest_v, c.cost, c.group)
                self.update_cl(cl, c)


        #return sol
    def create_cl(self):
        instance = self.instance
        n = instance.n
        # Candidate List of nodes
        cl = []
        for v in range(0, n):
            if v in self.selected_list:
                continue
            for k in range(0, self.groups):
                v_min, min_dist = self.distance_to(v, k)
                if min_dist > self.max_of * self.coef_bound:
                    c = Candidate(v, v_min, min_dist, k)
                    cl.append(c)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menor
        return cl

    def update_cl(self, cl, last_added):
        instance = self.instance
        to_remove = set()
        dict_last = instance.distance[last_added.v]
        for c in cl:
            if c.cost < self.max_of * self.coef_bound or c.v in self.selected_list or self.n_selected[c.group] == self.p:
                to_remove.add(c)
            elif c.group == last_added.group:
                d_to_last = dict_last[c.v]
                if d_to_last < self.max_of * self.coef_bound:
                    to_remove.add(c)
                elif d_to_last < c.cost:
                    c.cost = d_to_last
                    c.closest_v = last_added.v
        cl[:] = [c for c in cl if c not in to_remove]
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos

    def construct_solution_chained(self, beta_0, beta_1, k):

        stop = True
        while stop:
            index_selected = int((np.log(np.random.random()) / np.log(1 - beta_0))) % len(self.sorted_distances)
            edge = self.sorted_distances[index_selected]
            if (edge.v1 not in self.selected_list and edge.v2 not in self.selected_list):
                self.add(edge.v1, k)
                self.add(edge.v2, k)
                self.update_of(edge.v1, edge.v2, edge.distance, k)
                stop = False
            self.sorted_distances.pop(index_selected)

        cl = self.create_cl_chained(k)

        # 0 dado que es el greedy
        while len(self.selected_dict[k]) < self.p:
            if len(cl) < 1:
                self.of = 0
                self.end_iteration = True
                break
            if self.group_last_selection is not None:
                index_selected = self.find_first_with_group(cl, self.group_last_selection)
                self.group_last_selection = None
            else:
                index_selected = int((np.log(np.random.random()) / np.log(1 - beta_1))) % len(cl)

            if index_selected is None:
                self.of = 0
                self.end_iteration = True
                break

            else:
                c = cl.pop(index_selected)
                self.add(c.v, c.group)
                self.update_of(c.v, c.closest_v, c.cost, k)
                self.update_cl_chained(cl, c)

        # return sol

    def create_cl_chained(self, k):
        instance = self.instance
        n = instance.n
        # Candidate List of nodes
        cl = []
        for v in range(0, n):
            if v in self.selected_list:
                continue
            v_min, min_dist = self.distance_to(v, k)
            if min_dist > self.max_of * self.coef_bound:
                c = Candidate(v, v_min, min_dist, k)
                cl.append(c)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menor
        return cl

    def update_cl_chained(self, cl, last_added):
        instance = self.instance
        to_remove = set()
        dict_last = instance.distance[last_added.v]
        for c in cl:
            if c.cost < self.max_of * self.coef_bound or c.v in self.selected_list:
                to_remove.add(c)
            elif c.group == last_added.group:
                d_to_last = dict_last[c.v]
                if d_to_last < self.max_of * self.coef_bound:
                    to_remove.add(c)
                elif d_to_last < c.cost:
                    c.cost = d_to_last
                    c.closest_v = last_added.v
        cl[:] = [c for c in cl if c not in to_remove]
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos


    def distance_to(self, v, k, exclude_list=[]):
        min_dist = self.instance.sorted_distances[0].distance * 10
        v_min = -1
        for s in self.selected_dict[k]:
            if s not in exclude_list:
                d = self.instance.distance[s, v]
                if d < min_dist:
                    min_dist = d
                    v_min = s
        return v_min, min_dist

    def find_group_by_value(self, value):
        for key, values in self.selected_dict.items():
            if value in values:
                return key
        return None

    def run_exchage_LS(self, max_ls, active_ls3):

        # First look up all the critical nodes.

        n = self.instance.n

        for _ in range(max_ls):
            critical_nodes = {}
            critical_found = False
            for group, list_nodes in self.selected_dict.items():
                for i in range(self.p - 1):
                    for j in range(i + 1, self.p):
                        if self.instance.distance[list_nodes[i], list_nodes[j]] == self.of:
                            critical_nodes.setdefault(list_nodes[i], group)
                            critical_nodes.setdefault(list_nodes[j], group)
                            critical_found = True
                            break
                    if critical_found:
                        break
                if critical_found:
                    break

            blacklist = set()
            for c_node, group in critical_nodes.items():
                if c_node in blacklist:
                    continue
                improved = False

                for num1 in range(n):
                    v = self.integers_list[num1]
                    if v == c_node:
                        continue
                    if self.distance_to(v, group, [c_node])[1] > self.of:
                        # If the node owns to a group, check if the node can be changed
                        if v in self.selected_list:
                            v_group = self.find_group_by_value(v)
                            if v_group is not None:
                                # Try an exchange, else, try 3-exchange
                                if self.distance_to(c_node, v_group, [v])[1] > self.of:
                                    self.selected_dict[group].remove(c_node)
                                    self.selected_dict[v_group].remove(v)
                                    self.selected_list.remove(c_node)
                                    self.selected_list.remove(v)
                                    self.add(c_node, v_group)
                                    self.update_of(0, c_node, 0, v_group, recalculate=True)
                                    self.add(v, group)
                                    self.update_of(0, v, 0, group, recalculate=True)
                                    self.historial.append(self.of)
                                    blacklist.update([c_node, v])
                                    improved = True
                                    break
                                else:
                                    if active_ls3:
                                        for num2 in range(n):
                                            v2 = self.integers_list[num2]
                                            if v2 != c_node and v2 != v:
                                                if v2 in self.selected_list:
                                                    continue
                                                else:
                                                    if self.distance_to(v2, v_group, [v])[1] > self.of:
                                                        self.selected_dict[group].remove(c_node)
                                                        self.selected_dict[v_group].remove(v)
                                                        self.selected_list.remove(c_node)
                                                        self.add(v, group)
                                                        self.update_of(0, v, 0, group, recalculate=True)
                                                        self.add(v2, v_group)
                                                        self.update_of(0, v2, 0, v_group, recalculate=True)
                                                        self.historial.append(self.of)
                                                        blacklist.update([c_node, v, v2])

                                                        improved = True
                                                        break
                                    else:
                                        continue
                                    if improved:
                                        break
                        else:
                            self.selected_dict[group].remove(c_node)
                            self.selected_list.remove(c_node)
                            self.add(v, group)
                            self.update_of(0, v, 0, group, recalculate=True)
                            self.historial.append(self.of)
                            blacklist.update([c_node, v])
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break

    def run_exchage_LS_chained(self, max_ls, k):

        # First look up all the critical nodes.

        n = self.instance.n

        for _ in range(max_ls):
            critical_nodes = {}
            critical_found = False
            for i in range(len(self.selected_dict[k])):
                for j in range(i + 1, self.p):
                    if self.instance.distance[self.selected_dict[k][i], self.selected_dict[k][j]] == self.of:
                        critical_nodes.setdefault(self.selected_dict[k][i], k)
                        critical_nodes.setdefault(self.selected_dict[k][j], k)
                        critical_found = True
                        break
                if critical_found:
                    break


            blacklist = set()
            for c_node, group in critical_nodes.items():
                if c_node in blacklist:
                    continue
                improved = False

                for num1 in range(n):
                    v = self.integers_list[num1]
                    if v in self.selected_list:
                        continue
                    if self.distance_to(v, group, [c_node])[1] > self.of:
                        # If the node owns to a group, check if the node can be changed
                        self.selected_dict[group].remove(c_node)
                        self.selected_list.remove(c_node)
                        self.add(v, group)
                        self.update_of_chained(0, v, 0, group, recalculate=True)
                        self.historial.append(self.of)
                        blacklist.update([c_node, v])
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    def update_of_chained(self, v_min1, v_min2, of, group, recalculate= False):
        if recalculate:
            new_of = 999999999
            for v1 in self.selected_dict[group]:
                for v2 in self.selected_dict[group]:
                    if v1 > v2:
                        dist = self.instance.distance[v1][v2]
                        if new_of > dist:
                            new_of = dist
                            new_v1_min = v1
                            new_v2_min = v2
            self.dict_disp_group[group] = [(new_v1_min, new_v2_min), new_of]
            self.of = new_of

        else:
            if len(self.dict_disp_group[group]) == 0:
                self.dict_disp_group[group] = [(v_min1, v_min2), of]

            elif self.dict_disp_group[group][1] > of:
                self.dict_disp_group[group] = [(v_min1, v_min2), of]

            if self.of > of:
                self.of = of
                self.v_min1 = v_min1
                self.v_min2 = v_min2

    def anti_clustering(self):
        import numpy as np
        from scipy.cluster.hierarchy import linkage, fcluster
        from pyclustering.cluster.kmedoids import kmedoids
        from pyclustering.utils import calculate_distance_matrix
        from scipy.spatial.distance import squareform

        def hierarchical_clustering(distance_matrix, n_clusters):
            # Perform hierarchical clustering
            Z = linkage(squareform(distance_matrix), method='ward')
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            return labels

        def get_initial_medoids(labels, distance_matrix):
            medoids = []
            unique_labels = np.unique(labels)
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                sub_matrix = distance_matrix[cluster_indices][:, cluster_indices]
                medoid_index = cluster_indices[np.argmin(np.sum(sub_matrix, axis=0))]
                medoids.append(medoid_index)
            return medoids

        def k_medoids_clustering(distance_matrix, initial_medoids, n_clusters):
            kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
            kmedoids_instance.process()
            clusters = kmedoids_instance.get_clusters()
            return clusters

        def balance_clusters(clusters, n_clusters, cluster_size):
            flattened_clusters = [item for sublist in clusters for item in sublist]
            balanced_clusters = []
            for i in range(n_clusters):
                balanced_clusters.append(flattened_clusters[i * cluster_size: (i + 1) * cluster_size])
            return balanced_clusters

        def calculate_min_distances(distance_matrix, clusters):
            max_distances = []
            for cluster in clusters:
                cluster_distances = 1/distance_matrix[np.ix_(cluster, cluster)]
                max_distance = np.min(cluster_distances)
                max_distances.append(max_distance)
            return max_distances

        def clustering(distance_matrix, n_clusters):
            n_points = len(distance_matrix)
            cluster_size = n_points // n_clusters

            # Step 1: Initial Clustering using Hierarchical Clustering
            labels = hierarchical_clustering(distance_matrix, n_clusters)

            # Step 2: Get Initial Medoids
            initial_medoids = get_initial_medoids(labels, distance_matrix)

            # Step 3: Refine Clusters using K-Medoids
            clusters = k_medoids_clustering(distance_matrix, initial_medoids, n_clusters)

            # Step 4: Balance Clusters
            balanced_clusters = balance_clusters(clusters, n_clusters, cluster_size)

            return balanced_clusters

        # Turn
        distance_matrix = np.zeros((self.p, self.p))
        for i in range(self.p):
            matrix_i = self.distance[self.selected_list[i]]
            for j in range(self.p):
                if i==j:
                    distance_matrix[i, j] = 0
                else:
                    if (matrix_i[j] == 0):
                        distance_matrix[i, j] = 1000000
                        distance_matrix[j, i] = 1000000
                        ez
                    distance_matrix[i, j] = 1/matrix_i[j]
                    distance_matrix[j, i] = 1/matrix_i[j]
        print(distance_matrix)


        n_clusters = 2
        clusters = clustering(distance_matrix, n_clusters)

        # Calculate the maximum distance within each cluster
        min_distances = calculate_min_distances(distance_matrix, clusters)

        # Determine the minimum of these maximum distances
        min_max_distance = min(min_distances)

        #print(f"Clusters: {clusters}")
        #print(f"Minimum distances within each cluster: {min_distances}")
        #print(f"Minimum of the maximum distances: {min_max_distance}")
        print(self.of)




