import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import time
from gurobipy import *
class Solution_Gurobi:
    def __init__(self, param_dict, max_time):
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


        """"
        self.weight = weight
        self.max_capacity = -1
        self.max_min_dist = -1
        self.real_alpha = real_alpha
        # Dynamic enviroment
        random.seed(t.seed+count)
        """
    def run_algorithm(self):
        # F1
        self.construct_solution_kgpdp(k=self.groups, p=self.p, time_max=self.max_time)
        # F2
        self.construct_solution_kgpdp_compact(k=self.groups, p=self.p, time_max=self.max_time)

        # F3 Step by Step
        bound = 0.1
        start = time.time()
        improved = True

        while improved:
            improved = False
            current_time = time.time() - start
            if (self.max_time - current_time > 1):
                of = self.construct_solution_kgpdp_packing(k=self.groups, p=self.p, time_max=self.max_time - current_time, bound=next_bound, mode="Sbs")
                if of > bound:
                    improved = True
                    bound = of
        self.save_dict_to_txt('output/outputPackingSbS.txt', bound, self.instance_name, "PackingSbS", time.time() - start)

        #F3 Binary
        sorted_distances = list(dict.fromkeys([i.distance for i in self.sorted_distances]))
        down_index = len(sorted_distances)
        up_index = 0
        start = time.time()
        loop = True
        best_of = 0
        while loop:
            improved = False
            current_time = time.time() - start
            if (self.max_time - current_time > 1):
                target_index = int(up_index + (down_index - up_index)/2)
                if target_index == up_index:
                    loop = False
                    break
                bound = sorted_distances[target_index]
                of = self.construct_solution_kgpdp_packing(k=self.groups, p=self.p, time_max=self.max_time - current_time, bound=bound, mode="Binary")
                if of > 0:
                    bound = of
                    best_of = of
                    down_index = target_index
                else:
                    up_index = target_index
            else:
                break


        self.save_dict_to_txt('output/outputPackingBinary.txt', best_of, self.instance_name, "PakingBinary", time.time() - start)

        return self.of

    def evalSolution(self,distance, sol):
        minDist = 0x3f3f3f
        n = len(sol)
        for i in range(n):
            ii, it = sol[i]  # Extract the first pair (i, t1)
            for j in range(i + 1, n):
                jj, jt = sol[j]  # Extract the second pair (j, t2)
                if it == jt:
                    minDist = min(minDist, distance[ii,jj])
        return minDist


    def run_algorithm_chained(self):
        # Sayah
        selected_list = []
        max_time = int(self.max_time/self.groups)
        for k in range(self.groups):
            print(self.instance_name + " " + str(k))
            indices_to_remove = self.construct_solution_kgpdp_compact_chained(p=self.p,
                                                                         time_max=max_time)

            # Remove rows
            self.distance = [row for i, row in enumerate(self.distance) if i not in indices_to_remove]

            # Remove columns
            self.distance = [[elem for j, elem in enumerate(row) if j not in indices_to_remove] for row in self.distance]


        return self.of
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

        # return sol
    def construct_solution_kgpdp(self, k, p, time_max):

        instance = self.instance
        n = instance.n

        model = pyo.ConcreteModel()

        model.i = RangeSet(0, n)
        model.k = RangeSet(0, k)

        model.X = pyo.Var(model.i, model.k, within=Binary)

        model.d = pyo.Var(bounds=(0, None))

        X = model.X

        d = model.d
        M = np.max(self.distance)

        model.C1 = pyo.ConstraintList()
        for ki in range(k):
            x_sum = sum([X[i, ki] for i in range(n)])
            model.C1.add(expr= x_sum == p)

        model.C2 = pyo.ConstraintList()

        for i in range(n):
            x_sum = sum([X[i, ki] for ki in range(k)])
            model.C2.add(expr=x_sum <= 1)

        model.C3 = pyo.ConstraintList()
        for i in range(n-1):
            for j in range(i+1,n):
                for ki in range(k):
                    model.C3.add(expr= M*X[i, ki] + M*X[j, ki] + d <= 2*M + self.distance[i, j] )


        model.obj = pyo.Objective(expr=d, sense=maximize)

        print("Model Built")
        opt = SolverFactory('gurobi')

        results = opt.solve(model)
        d_value = pyo.value(d)

        time_value = self.extract_time_from_string(str(results["Solver"]))
        self.save_dict_to_txt('output/outputModels.txt', d_value, self.instance_name, "kuby", time_value)


        solution_dict = {k: [] for k in range(0, k)}
        for k in range(0, k):
            for i in range(0, n):
                if pyo.value(X[i, k]) > 0:
                    solution_dict[k].append(i)


    def construct_solution_kgpdp_compact(self, k, p, time_max):
        instance = self.instance
        n = instance.n
        model = pyo.ConcreteModel()

        Dm = np.max(self.distance)
        sorted_distances = list(dict.fromkeys([i.distance for i in self.sorted_distances]))
        sorted_distances.reverse()
        model.i = RangeSet(0, n)
        model.k = RangeSet(0, k)
        model.M = RangeSet(0, len(sorted_distances))

        model.X = pyo.Var(model.i, model.k, within=Binary)

        model.u = pyo.Var(model.M, within=Binary)

        X = model.X

        u = model.u

        model.C1 = pyo.ConstraintList()
        for ki in range(k):
            x_sum = sum([X[i, ki] for i in range(n)])
            model.C1.add(expr= x_sum == p)

        model.C2 = pyo.ConstraintList()

        for i in range(n):
            x_sum = sum([X[i, ki] for ki in range(k)])
            model.C2.add(expr=x_sum <= 1)

        model.C3 = pyo.ConstraintList()
        for i in range(n-1):
            for j in range(i+1,n):
                index = sorted_distances.index(self.distance[i, j])
                for ki in range(k):
                    model.C3.add(expr=  X[i, ki] + X[j, ki] <= 1 + u[index])

        model.C4 = pyo.ConstraintList()
        for m in range(1, len(sorted_distances)):
            model.C4.add(expr= u[m-1] <= u[m])


        telescopic_sum = 0
        for m in range(0, len(sorted_distances)-1):
            telescopic_sum += u[m] * (sorted_distances[m+1] - sorted_distances[m])
        model.obj = pyo.Objective(expr=Dm - telescopic_sum , sense=maximize)

        print("Model Built")
        opt = SolverFactory('gurobi')
        opt.options['TimeLimit'] = time_max
        results = opt.solve(model)



        X_value = [[i, ki] for i in range(n) for ki in range(k) if pyo.value(X[i, ki]) > 0]
        solution = []
        for m in range(0, len(sorted_distances)):
            if pyo.value(u[m]) > 0:
                solution.append(sorted_distances[m])
                time_value = self.extract_time_from_string(str(results["Solver"]))
                self.save_dict_to_txt('output/outputModels.txt', sorted_distances[m], self.instance_name, "sayah", time_value)
                break


    def construct_solution_kgpdp_compact_chained(self, p, time_max):
        # pdp
        instance = self.instance
        n = len(self.distance)
        model = pyo.ConcreteModel()

        Dm = np.max(self.distance)
        sorted_distances = list(dict.fromkeys([i.distance for i in self.sorted_distances]))
        sorted_distances.reverse()
        model.i = RangeSet(0, n)
        model.M = RangeSet(0, len(sorted_distances))

        model.X = pyo.Var(model.i, within=Binary)

        model.u = pyo.Var(model.M, within=Binary)


        X = model.X

        u = model.u

        model.C1 = pyo.ConstraintList()
        x_sum = sum([X[i] for i in range(n)])
        model.C1.add(expr=x_sum == p)

        model.C3 = pyo.ConstraintList()
        for i in range(n - 1):
            for j in range(i + 1, n):
                index = sorted_distances.index(self.distance[i][j])
                model.C3.add(expr=X[i] + X[j] <= 1 + u[index])

        model.C4 = pyo.ConstraintList()
        for m in range(1, len(sorted_distances)):
            model.C4.add(expr=u[m - 1] <= u[m])


        telescopic_sum = 0
        for m in range(0, len(sorted_distances) - 1):
            telescopic_sum += u[m] * (sorted_distances[m + 1] - sorted_distances[m])
        model.obj = pyo.Objective(expr=Dm - telescopic_sum, sense=maximize)

        opt = SolverFactory('gurobi')
        opt.options['TimeLimit'] = time_max
        results = opt.solve(model)

        solution = []
        for m in range(0, len(sorted_distances)):
            if pyo.value(u[m]) > 0:
                solution.append(sorted_distances[m])
                print(solution)
                time_value = self.extract_time_from_string(str(results["Solver"]))
                self.save_dict_to_txt('output/outputChained.txt', sorted_distances[m], self.instance_name, "Chained", time_value)
                break

        selected_list = [i for i in range(len(self.distance)) if pyo.value(X[i]) > 0]

        return selected_list

    def find_last_more_than_bound(self, sorted_list, bound):
        for i in range(len(sorted_list)):
            if sorted_list[i] <= bound:
                return sorted_list[i-1]
    def construct_solution_kgpdp_packing(self, k, p, time_max, bound, mode):
        #pdp
        instance = self.instance
        n = instance.n
        model = pyo.ConcreteModel()


        sorted_distances = list(dict.fromkeys([i.distance for i in self.sorted_distances]))
        if mode == "SbS":
            problem_bound = self.find_last_more_than_bound(sorted_distances, bound)
        else:
            problem_bound = bound

        sorted_distances.reverse()

        model.i = RangeSet(0, n)
        model.k = RangeSet(0, k)

        model.X = pyo.Var(model.i, model.k, within=Binary)

        X = model.X


        model.C1 = pyo.ConstraintList()
        for ki in range(k):
            x_sum = sum([X[i, ki] for i in range(n)])
            model.C1.add(expr= x_sum == p)

        model.C2 = pyo.ConstraintList()

        for i in range(n):
            x_sum = sum([X[i, ki] for ki in range(k)])
            model.C2.add(expr=x_sum <= 1)

        model.C3 = pyo.ConstraintList()
        for i in range(n-1):
            for j in range(i+1,n):

                index = sorted_distances.index(self.distance[i, j])
                if self.distance[i, j] < problem_bound:
                    for ki in range(k):
                        model.C3.add(expr= X[i, ki] + X[j, ki] <= 1)
        model.obj = pyo.Objective(expr=1 , sense=maximize)

        opt = SolverFactory('gurobi')

        opt.options['TimeLimit'] = time_max
        try:
            results = opt.solve(model)
            if results.solver.termination_condition != 'infeasible':
                min_distance = 10000000
                for ki in range(k):
                    for i in range(n - 1):
                        if pyo.value(X[i, ki]) > 0.5:
                            for j in range(i + 1, n):
                                if pyo.value(X[j, ki]) > 0.5:
                                    distance = self.distance[i, j]
                                    if distance < min_distance:
                                        min_distance = distance
                print("Current Sol: ", min_distance)
                solution = min_distance
                return solution
            else:
                print("Infeasible")
                return 0
        except Exception as e:
            return 0
