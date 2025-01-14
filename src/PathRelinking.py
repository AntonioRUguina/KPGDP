import copy
class PathRelinking:
    def __init__(self, param_dict, set_solutions):
        self.instance = param_dict["inst"]
        self.distance = self.instance.distance
        self.groups = self.instance.k
        self.p = self.instance.p
        self.set_solutions = set_solutions
        self.matches = {}
        self.selected_list = []
        self.selected_dict = {}
        self.of = 0
        self.dict_disp_group = {}
        self.historial = []

    def run_algorithm(self):
        len_set = len(self.set_solutions)
        list_of = []

        index = 0

        for indx1 in range(len_set - 1):
            for indx2 in range(indx1 + 1, len_set):
                print(indx1, indx2)
                pr_dict = []
                sol1 = self.set_solutions[indx1]
                sol2 = self.set_solutions[indx2]

                self.match_groups_min(sol1, sol2)
                nodes_sol1, common_elements = self.distribute_elements(sol1, sol2)
                dict_of = self.scan_of(sol1)

                stop = False
                while not stop:
                    reward_dict = {}
                    dict_of = self.scan_of(sol1)
                    for group, nodes in sol1.items():
                        candidates = [i for i in sol2[self.matches[group]] if i not in common_elements[group]]

                        if candidates:
                            for node in (i for i in nodes if i not in common_elements[group]):
                                distance_node = self.distance_to_solution(node, group, sol1, exclude_list=[node])
                                for candidate in candidates:
                                    already_in_solution = candidate in nodes_sol1
                                    distance = self.distance_to_solution(candidate, group, sol1, exclude_list=[node])
                                    reward = distance - distance_node

                                    if (dict_of[group] ==min(dict_of.values())):
                                        reward = reward * 2

                                    group_candidate = None
                                    if already_in_solution:
                                        group_candidate = self.find_group(candidate, sol1)

                                        distance_candidate  = self.distance_to_solution(node, group_candidate, sol1,
                                                                                                       exclude_list=[candidate])
                                        distance2 = self.distance_to_solution(node, group_candidate, sol1,
                                                                     exclude_list=[candidate])
                                        reward += distance2 - distance_candidate
                                        if (dict_of[group_candidate] == min(dict_of.values())):
                                            reward += distance2 - distance_candidate

                                    reward_dict[(group, node, candidate, already_in_solution, group_candidate)] = reward

                    if not reward_dict:
                        stop = True
                        break

                    best_move = max(reward_dict.items(), key=lambda x: x[1])[0]
                    group, node, candidate, already_in_solution, group_candidate = best_move

                    sol1[group].remove(node)
                    sol1[group].append(candidate)

                    if already_in_solution:
                        sol1[group_candidate].remove(candidate)
                        sol1[group_candidate].append(node)

                    index += 1
                    dict_of = self.scan_of(sol1)
                    pr_dict.append({"index":index, "sum_of_groups": sum(dict_of.values()), "min_of": min(dict_of.values()), "solution": copy.deepcopy(sol1)})
                    nodes_sol1, common_elements = self.distribute_elements(sol1, sol2)

                #Take the best in terms of sum of dispersion groups
                pr_dict.sort(key=lambda x: x["sum_of_groups"], reverse = True)
                best_max_sum = pr_dict.pop(0)

                self.selected_dict = best_max_sum["solution"]
                combined_list = []
                for key in best_max_sum['solution']:
                    combined_list.extend(best_max_sum['solution'][key])
                self.selected_list = combined_list
                self.of = best_max_sum["min_of"]
                self.dict_disp_group = self.scan_of(best_max_sum['solution'])
                self.historial = []

                list_of.append(self.of)
                print("pre_ls sum",self.of)


                self.run_exchage_LS(100,True)

                list_of.append(self.of)
                print("post_ls sum",self.of)

                # Take the best in terms of min of dispersion groups

                pr_dict.sort(key=lambda x: x["min_of"], reverse=True)
                best_max = pr_dict.pop(0)

                self.selected_dict = best_max["solution"]
                combined_list = []
                for key in best_max['solution']:
                    combined_list.extend(best_max['solution'][key])
                self.selected_list = combined_list
                self.of = best_max["min_of"]
                self.dict_disp_group = self.scan_of(best_max['solution'])
                self.historial = []

                list_of.append(self.of)
                print("pre_ls maxmin", self.of)

                self.run_exchage_LS(100, True)

                list_of.append(self.of)
                print("post_ls maxmin", self.of)



        return max(list_of)

    def find_common_elements(self, list1, list2):
        return len(set(list1) & set(list2))

    def match_groups(self,sol1, sol2):
        # Create a result dictionary to store the matches
        matches = {}

        # Iterate over each list in the first dictionary
        for key1, list1 in sol1.items():
            max_common = -1
            best_match_key = None

            # Compare with each list in the second dictionary
            for key2, list2 in sol2.items():
                common_elements = self.find_common_elements(list1, list2)

                # Update the best match if the current pair has more common elements
                if common_elements > max_common and key2 not in matches.values():
                    max_common = common_elements
                    best_match_key = key2

            # Store the best match for the current list from the first dictionary
            matches[key1] = best_match_key
        self.matches = matches

    def match_groups_min(self, sol1, sol2):
        # Create a result dictionary to store the matches
        matches = {}

        # Iterate over each list in the first dictionary
        for key1, list1 in sol1.items():
            min_common = 1000
            best_match_key = None

            # Compare with each list in the second dictionary
            for key2, list2 in sol2.items():
                common_elements = self.find_common_elements(list1, list2)

                # Update the best match if the current pair has more common elements
                if common_elements < min_common and key2 not in matches.values():
                    min_common = common_elements
                    best_match_key = key2

            # Store the best match for the current list from the first dictionary
            matches[key1] = best_match_key
        self.matches = matches

    def distribute_elements(self, sol1, sol2):
        # Create a set to store all common elements (to avoid duplicates)
        common_elements = {}
        # Find the common elements for each matched pair and add to the set
        for key1, key2 in self.matches.items():
            common_elements[key1] = list(set(sol1[key1]) & set(sol2[key2]))

        # Create a set to store all unique elements
        all_unique_elements = set()

        # Add all elements from dict1 to the set
        for value_list in sol1.values():
            all_unique_elements.update(value_list)

        # Convert the set to a list
        return  list(all_unique_elements), common_elements

    def scan_of(self, sol):
        dict_of = {}
        for group, nodes in sol.items():
            of = 10000
            len_nodes = len(nodes)
            for idx1 in range(len_nodes-1):
                node_1 = nodes[idx1]
                matrix_1 = self.distance[node_1]
                for idx2 in range(idx1+1, len_nodes):
                    node_2 = nodes[idx2]
                    dist = matrix_1[node_2]
                    if dist < of:
                        of = dist
            dict_of[group] = of
        return dict_of

    def distance_to_solution(self, v, k, sol, exclude_list=[]):
        min_dist = self.instance.sorted_distances[0].distance * 10
        v_min = -1
        for s in sol[k]:
            if s not in exclude_list:
                d = self.instance.distance[s, v]
                if d < min_dist:
                    min_dist = d
                    v_min = s
        return min_dist
    def find_group(self,v,sol):
        for group, nodes in sol.items():
            if v in nodes:
                return group

    def run_exchage_LS(self, max_ls, active_ls3):

        # First look up all the critical nodes.

        n = self.instance.n
        total_time = 0
        for _ in range(max_ls):

            critical_nodes = {}
            critical_found = False
            for group, list_nodes in self.selected_dict.items():
                for i in range(self.p - 1):
                    matrix_i = self.distance[list_nodes[i]]
                    for j in range(i + 1, self.p):
                        if matrix_i[list_nodes[j]] == self.of:
                            critical_nodes.setdefault(list_nodes[i], group)
                            critical_nodes.setdefault(list_nodes[j], group)
                            critical_found = True
                            break
                    if critical_found:
                        break
                if critical_found:
                    break

            for c_node, group in critical_nodes.items():
                improved = False

                for v in range(n):
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
                                    improved = True
                                    self.historial.append(self.of)
                                    break
                                else:

                                    if active_ls3:
                                        for v2 in range(n):
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
                                                        improved = True
                                                        self.historial.append(self.of)
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
                            improved = True
                            self.historial.append(self.of)
                            break
                if improved:
                    break
            if not improved:
                break

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
    def add(self, v, k):
        self.selected_dict[k].append(v)
        self.selected_list.append(v)

    def find_group_by_value(self, value):
        for key, values in self.selected_dict.items():
            if value in values:
                return key
        return None

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
            self.dict_disp_group[group] = new_of
            self.of = min([self.dict_disp_group[k] for k in range(self.groups)])

