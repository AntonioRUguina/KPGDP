import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
class Solution_Gurobi:
    def __init__(self, param_dict, max_time, list_of_selected_solution, grasp_bound):
        self.instance = param_dict["inst"]
        self.instance_name = param_dict['t'].instName
        self.delta = param_dict["delta"]
        self.sorted_distances = self.instance.sorted_distances
        self.distance = self.instance.distance
        self.selected_list = []
        self.selected_dict = {i: [] for i in range(0, self.instance.k)}
        self.n_selected = self.instance.k*[0]
        self.v_min1 = -1
        self.v_min2 = -1
        self.of = self.sorted_distances[0].distance * 10
        self.capacity = 0
        self.time = []
        self.groups = self.instance.k
        self.patron = []
        self.p = self.instance.p
        self.max_time = max_time
        self.list_of_selected_solution = list_of_selected_solution
        self.grasp_bound = grasp_bound


    def run_algorithm(self):
        uniqueList = list(set([item for sublist in self.list_of_selected_solution for item in sublist]))
        of = self.construct_solution_kgpdp_binary_PR(k=self.groups, p=self.p, time_max=self.max_time, uniqueList = uniqueList, grasp_bound=self.grasp_bound)
        return of

    def find_last_more_than_bound(self, sorted_list, bound):
        for i in range(len(sorted_list)):
            if sorted_list[i] <= bound:
                return sorted_list[i-1]
    def extract_time_from_string(self,text):
        try:
            # Split the text by lines
            lines = text.split('\n')
            # Iterate over each line
            for line in lines:
                # Check if the line starts with "Time:"
                if line.startswith("  Time:"):
                    # Split the line by ":" and get the second part
                    time_value = line.split(":")[1].strip()
                    return time_value
            print("Value after 'Time:' not found in the input string")
            return None
        except Exception as e:
            print("An error occurred:", str(e))
            return None
    def save_dict_to_txt(self,filename, d_value, instance_name,model, time):
        with open(filename, 'a') as file:
                file.write(f"{instance_name} {model} {time}: {d_value}\n")

    def construct_solution_kgpdp_binary_PR(self, k, p, time_max, uniqueList, grasp_bound):
        #pdp
        instance = self.instance
        n = instance.n
        model = pyo.ConcreteModel()


        sorted_distances = list(dict.fromkeys([i.distance for i in self.sorted_distances if (i.v1 in uniqueList and i.v2 in uniqueList)]))
        bound = self.find_last_more_than_bound(sorted_distances,grasp_bound)
        sorted_distances.reverse()
        Dm = np.max(sorted_distances)

        model.i = RangeSet(0, len(uniqueList))
        model.k = RangeSet(0, k)
        model.M = RangeSet(0, len(sorted_distances))

        model.X = pyo.Var(model.i, model.k, within=Binary)

        X = model.X

        l = len(uniqueList)

        model.C1 = pyo.ConstraintList()
        for ki in range(k):
            x_sum = sum([X[i, ki] for i in range(l)])
            model.C1.add(expr= x_sum == p)
        print("c1 Built")

        model.C2 = pyo.ConstraintList()

        for i in range(l):
            x_sum = sum([X[i, ki] for ki in range(k)])
            model.C2.add(expr=x_sum <= 1)
        print("c2 Built")

        model.C3 = pyo.ConstraintList()
        for i in range(l-1):
            for j in range(i+1,l):
                associated_i = uniqueList[i]
                associated_j = uniqueList[j]
                index = sorted_distances.index(self.distance[associated_i, associated_j])
                if self.distance[associated_i, associated_j] < bound:
                    for ki in range(k):
                        model.C3.add(expr= X[i, ki] + X[j, ki] <= 1)
        print("c3 Built")
        model.obj = pyo.Objective(expr=1 , sense=maximize)

        print("Model Built")
        opt = SolverFactory('gurobi')

        opt.options['TimeLimit'] = time_max
        # If there are memory problems, then reduce the Threads
        #opt.options['Threads'] = 1
        try:
            results = opt.solve(model)
            print(results)

            if results.solver.termination_condition != 'infeasible':
                min_distance = 10000000
                for ki in range(k):
                    for i in range(l - 1):
                        if pyo.value(X[i, ki]) > 0.5:
                            for j in range(i + 1, l):
                                if pyo.value(X[j, ki]) > 0.5:
                                    associated_i = uniqueList[i]
                                    associated_j = uniqueList[j]
                                    distance = self.distance[associated_i, associated_j]
                                    if distance < min_distance:
                                        min_distance = distance
                print("PR SOLUTION: ", min_distance)
                solution = min_distance
                #print('X:', X_value)
                return solution
            else:
                print("Infeasible")
                return 0
        except Exception as e:
            return 0
            # print(e)