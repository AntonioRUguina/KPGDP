import random
from Utils.objects import (Candidate)
import numpy as np


class Solution_Group:
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
        self.coef_bound = 0.75
        self.integers_list = []
        self.end_iteration = False
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
        self.historial.append(self.of)
        max_ls = 150

        #if not self.end_iteration:
        #    self.run_exchage_LS(max_ls, self.active_ls3)

        return self.of

    def save_dict_to_txt(self, filename, d_value, instance_name, model, time, seed):
        with open(filename, 'a') as file:
                file.write(f"{instance_name} {seed} {model} {time}: {d_value}\n ")

    def update_of(self, v_min1, v_min2, of, group, recalculate= False):
        if recalculate:
            new_of = 999999999
            for v1 in self.selected_dict[group]:
                matrix_v1 = self.instance.distance[v1]
                for v2 in self.selected_dict[group]:
                    if v1 > v2:
                        dist = matrix_v1[v2]
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

        cl_groups = self.create_cl_groups()
        # 0 dado que es el greedy
        for it in range(self.p-2):
            for k in range(0, self.groups):
                cl = cl_groups[k]
                candidate_ok = False
                index_selected = 0
                while not candidate_ok and not self.end_iteration:
                    if len(cl) < 1:
                        self.of = 0
                        self.end_iteration = True
                        break
                    if it == self.p-1:
                        index_selected = self.find_first_with_group(cl, k)
                    else:
                        index_selected = int((np.log(np.random.random()) / np.log(1 - beta_1))) % len(cl)

                    if index_selected is None:
                        self.of = 0
                        self.end_iteration = True
                        break
                    else:
                        if cl[index_selected].v not in self.selected_list:
                            candidate_ok = True
                        else:
                            cl.pop(index_selected)
                if not self.end_iteration:
                    c = cl.pop(index_selected)
                    self.add(c.v, c.group)
                    self.update_of(c.v, c.closest_v, c.cost, c.group)
                    self.update_cl(cl, c)

        #return sol

    def create_cl_groups(self):
        instance = self.instance
        n = instance.n
        # Candidate List of nodes
        cl = {i : [] for i in range(self.groups)}
        for k in range(0, self.groups):
            for v in range(0, n):
                if v in self.selected_list:
                    continue
                v_min, min_dist = self.distance_to(v, k)
                if min_dist > self.max_of * self.coef_bound:
                    c = Candidate(v, v_min, min_dist, k)
                    cl[k].append(c)
            cl[k].sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menor
        return cl

    def update_cl(self, cl, last_added):
        instance = self.instance
        to_remove = set()
        dict_last = instance.distance[last_added.v]
        for c in cl:
            if c.cost < self.max_of * self.coef_bound or c.v in self.selected_list or self.n_selected[c.group] == self.p:
                to_remove.add(c)
            else:
                d_to_last = dict_last[c.v]
                if d_to_last < self.max_of * self.coef_bound:
                    to_remove.add(c)
                elif d_to_last < c.cost:
                    c.cost = d_to_last
                    c.closest_v = last_added.v
        cl[:] = [c for c in cl if c not in to_remove]
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos

    def distance_to(self, v, k, exclude_list=[] ):
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
                            critical_found=True
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
                                    self.selected_list.append(v)
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
                                                        self.selected_list.append(v2)
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
                            self.selected_list.append(v)
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