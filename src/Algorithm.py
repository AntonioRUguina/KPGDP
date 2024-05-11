import random

from Utils.objects import (Candidate)
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
class Solution:
    def __init__(self, param_dict):
        self.instance = param_dict["inst"]
        self.delta = param_dict["delta"]
        self.sorted_distances = self.instance.sorted_distances.copy()
        self.distance = self.instance.distance
        self.selected_list = []
        self.selected_dict = {k: [] for k in range(0, self.instance.k)}
        self.n_selected = self.instance.k*[0]
        self.v_min1 = -1
        self.v_min2 = -1
        self.of = self.sorted_distances[0].distance * 10
        self.dict_disp_group = {k: [] for k in range(0, self.instance.k)}
        self.capacity = 0
        self.time = []
        self.groups = self.instance.k
        self.patron = []
        self.n_ls =0
        self.historial = []
        self.improved_ls_group = False
        self.p = self.instance.p


        """"
        self.weight = weight
        self.max_capacity = -1
        self.max_min_dist = -1
        self.real_alpha = real_alpha
        # Dynamic enviroment
        random.seed(t.seed+count)
        """

    def run_algorithm(self, beta_0, beta_1):
        self.construct_solution(beta_0, beta_1)
        max_ls = 100
        self.historial.append(self.of)
        #print(self.selected_list, self.of)
        # ls between groups

        for i in range(max_ls):
            k = min(self.dict_disp_group, key=lambda x: self.dict_disp_group[x][1])
            improved = self.LS_change_node(k)
            self.historial.append(self.of)
            if not improved:
                break
            self.n_ls += 1

        if self.groups > 1:
            for i in range(max_ls):
                k = min(self.dict_disp_group, key=lambda x: self.dict_disp_group[x][1])
                self.LS_change_group(k1=k)
                self.historial.append(self.of)

                if not improved:
                    break



        return self.of

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

    def is_feasible(self):
        feasible = True
        for key, value in self.selected_dict.items():
            feasible = feasible and len(value) >= self.p

        return feasible



    def construct_solution(self, beta_0, beta_1):
        """
        original Heuristic
        Constructive heuristic (Deterministic Version)
        :return:
        """
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
        real_alpha = 0
        while not self.is_feasible():
            #distance_limit = cl[0].cost - (real_alpha * cl[len(cl) - 1].cost)
            index_selected = int((np.log(np.random.random()) / np.log(1 - beta_1))) % len(cl)

            """
            if len(self.selected_list) < 10:
                index_selected = int((np.log(np.random.random()) / np.log(1 - 0.6))) % len(cl)
            elif len(self.selected_list) < 30:
                index_selected = int((np.log(np.random.random()) / np.log(1 - 0.2))) % len(cl)
            else:
                index_selected = int((np.log(np.random.random()) / np.log(1 - 0.8))) % len(cl)
            """

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
                c = Candidate(v, v_min, min_dist, k)
                cl.append(c)
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menor
        return cl

    def update_cl(self, cl, last_added):
            instance = self.instance
            to_remove = []
            for c in cl:
                if c.v in self.selected_list:
                    to_remove.append(c)
                elif self.n_selected[c.group] == self.p:
                    to_remove.append(c)
                elif c.group == last_added.group:
                    d_to_last = instance.distance[last_added.v][c.v]
                    if d_to_last < c.cost:
                        c.cost = d_to_last
                        c.closest_v = last_added.v
            for c in to_remove:
                cl.remove(c)
            cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos

    def distance_to(self, v, k, exclude_list= [] ):
        min_dist = self.instance.sorted_distances[0].distance * 10
        v_min = -1
        for s in self.selected_dict[k]:
            if s not in exclude_list:
                d = self.instance.distance[s, v]
                if d < min_dist:
                    min_dist = d
                    v_min = s
        return v_min, min_dist



    def LS_change_node(self, k):
        min_nodes = self.dict_disp_group[k][0]
        nodes_list = [i for i in min_nodes]
        n = self.instance.n
        group_of = self.dict_disp_group[k][1]
        improved = False
        for node in nodes_list:
            for v in range(0, n):
                if v in self.selected_list:
                    continue
                else:
                    v_min, min_dist = self.distance_to(v, k, exclude_list=[node])
                    if min_dist > group_of:
                        self.selected_dict[k].remove(node)
                        self.selected_list.remove(node)
                        self.add(v, k)
                        self.update_of(v_min, v, min_dist, k, recalculate=True)
                        group_of = min_dist
                        improved = True
                        break
            if improved:
                break
        return improved

    def LS_change_group(self, k1):
        min_nodes_k1 = self.dict_disp_group[k1][0]
        nodes_list_k1 = [i for i in min_nodes_k1]
        for k2 in range(self.groups):
            if k1 != k2:
                for node_k1 in nodes_list_k1:
                    for node_k2 in self.selected_dict[k2]:
                        k1_min_node, k1_value = self.distance_to(node_k1, k2, exclude_list=[node_k2])
                        k2_min_node, k2_value = self.distance_to(node_k2, k1, exclude_list=[node_k1])
                        if k1_value > self.of and k2_value > self.of:
                            self.selected_dict[k1].remove(node_k1)
                            self.selected_dict[k2].remove(node_k2)
                            self.add(node_k1, k2)
                            self.update_of(k1_min_node, node_k1, k1_value, k2, recalculate=True)
                            self.add(node_k2, k1)
                            self.update_of(k2_min_node, node_k2, k2_value, k1, recalculate=True)
                            self.improved_ls_group = True
                            return True


    def find_first_common_element_index(self, list1, list2):
        for index, element in enumerate(list1):
            if element in list2:
                return index
        return None

    def find_first_common_element_groups(self, list1, list2):
        for index, element in enumerate(list1):
            if element[0] in list2[element[1]]:
                return index
        return None


    def analizar_patron(self, list_groups):

        for k in range(0, self.groups):
            list = list_groups[k]
            i = 0
            stop = True
            while stop:
                index_selected = i
                edge = self.sorted_distances[index_selected]
                if (edge.v1 in list and edge.v2 in list and
                        (edge.v1 not in self.selected_list and edge.v2 not in self.selected_list)):
                    self.add(edge.v1, k)
                    self.add(edge.v2, k)
                    self.update_of(edge.v1, edge.v2, edge.distance, k)
                    stop = False
                    self.patron.append(index_selected)
                i += 1

        cl = self.create_cl()

        real_alpha = 0
        while len(self.selected_list) < self.p*2:
            v_values = [[c.v, c.group] for c in cl]
            index_sol = self.find_first_common_element_groups(v_values, list_groups)
            self.patron.append(index_sol)
            c = cl.pop(index_sol)
            self.add(c.v, c.group)
            self.update_cl(cl, c)
            self.update_of(c.v, c.closest_v, c.cost, c.group)
        plt.plot(self.patron, marker='o')
        plt.show()
        print(self.of)

        # return sol

