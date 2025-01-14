import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
class Solution_Gurobi:
    def __init__(self, param_dict, max_time, uniqueList1, uniqueList2, grasp_bound):
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
        self.uniqueList1 = uniqueList1
        self.uniqueList2 = uniqueList2
        self.grasp_bound = grasp_bound


        """"
        self.weight = weight
        self.max_capacity = -1
        self.max_min_dist = -1
        self.real_alpha = real_alpha
        # Dynamic enviroment
        random.seed(t.seed+count)
        """
    def run_algorithm(self):
        uniqueList = list(set(self.uniqueList1 + self.uniqueList2))
        # of = self.construct_solution_kgpdp_compact_PR(k=self.groups, p=self.p, time_max=self.max_time, uniqueList = uniqueList, grasp_bound=self.grasp_bound)
        of = self.construct_solution_kgpdp_binary_PR(k=self.groups, p=self.p, time_max=self.max_time, uniqueList = uniqueList, grasp_bound=self.grasp_bound)
        return of

    def find_last_more_than_bound(self, sorted_list, bound):
        for i in range(len(sorted_list)):
            if sorted_list[i] <= bound:
                return sorted_list[i-1]
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

        # self.construct_solution_kgpdp_sum(k=self.groups, p= self.p, time_max = 3600)

        # print(self.selected_list, self.of)

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
    def construct_solution_kgpdp(self, k, p,time_max):

        #pdp
        instance = self.instance
        n = instance.n

        model = pyo.ConcreteModel()

        model.i = RangeSet(0, n)
        model.k = RangeSet(0, k)

        model.X = pyo.Var(model.i, model.k, within=Binary)

        model.d = pyo.Var(bounds=(0,None))


        #model.x = pyo.Var(range(n^2), within=Binary, bounds=(0, None))

        X = model.X

        d = model.d
        M = np.max(self.distance)

        model.C1 = pyo.ConstraintList()
        for ki in range(k):
            x_sum = sum([X[i, ki] for i in range(n)])
            model.C1.add(expr= x_sum == p)
        print("c1 Built")

        model.C2 = pyo.ConstraintList()

        for i in range(n):
            x_sum = sum([X[i, ki] for ki in range(k)])
            model.C2.add(expr=x_sum <= 1)
        print("c2 Built")

        model.C3 = pyo.ConstraintList()
        for i in range(n-1):
            for j in range(i+1,n):
                for ki in range(k):
                    model.C3.add(expr= M*X[i, ki] + M*X[j, ki] + d <= 2*M + self.distance[i, j] )

        print("c3 Built")

        model.obj = pyo.Objective(expr=d, sense=maximize)

        print("Model Built")
        opt = SolverFactory('gurobi')
        #opt = SolverFactory('gurobi_direct')
        #opt.options['tmlim'] = 10
        #opt.options['TimeLimit'] = time_max
        # If there are memory problems, then reduce the Threads
        opt.options['Threads'] = 1

        results = opt.solve(model)
        d_value = pyo.value(d)


        # Función para guardar el diccionario en un archivo de texto

        time_value = self.extract_time_from_string(str(results["Solver"]))
        # Guardar el diccionario en 'data.txt'
        self.save_dict_to_txt('output.txt', d_value, self.instance_name,"kuby", time_value)


        print('d:', d_value)
        #X_value = [pyo.value(X[i]) for i in range(n)]
        solution_dict = {k: [] for k in range(0, k)}
        for k in range(0, k):
            for i in range(0, n):
                if pyo.value(X[i, k]) > 0:
                    solution_dict[k].append(i)

        print (solution_dict)

        #return sol

    def construct_solution_kgpdp_compact_PR(self, k, p, time_max, uniqueList, grasp_bound):
        #pdp
        # uniqueList1 = [110, 9, 107, 38, 22, 25, 113, 86, 37, 111, 8, 35, 41, 18, 62, 40, 118, 43, 120, 134, 32, 21, 90, 143, 73, 13, 56, 42, 96, 72]
        # uniqueList2 =  [86, 110, 107, 25, 21, 13, 142, 9, 18, 63, 84, 38, 40, 62, 120, 43, 41, 102, 30, 60, 73, 113, 8, 37, 111, 143, 32, 90, 35, 22]
        # uniqueList = uniqueList1 + uniqueList2
        # uniqueList = [7,8,9,13,14,18,22,25,30,32,35,38,41,42,56,60,62,73,78,84,90,107,111,113]
        #uniqueList = range(150)
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

        model.u = pyo.Var(model.M, within=Binary)

        #model.x = pyo.Var(range(n^2), within=Binary, bounds=(0, None))

        X = model.X

        u = model.u

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
                for ki in range(k):
                    model.C3.add(expr=  X[i, ki] + X[j, ki] <= 1 + u[index])
        print("c3 Built")

        model.C4 = pyo.ConstraintList()
        for m in range(1, len(sorted_distances)):
            model.C4.add(expr= u[m-1] <= u[m])

        print("c4 Built")

        #model.C5 = pyo.ConstraintList()
        #for ki in range(k-1):
        #    i_sum = sum(i*X[i, ki] for i in range(n))
        #    i1_sum = sum(i*X[i, ki+1] for i in range(n))
        #    model.C5.add(expr= i_sum <= i1_sum)
        #print("c5 Built")

        #PR-CUT
        model.C5 = pyo.ConstraintList()
        telescopic_sum = 0

        for m in range(0, len(sorted_distances) - 1):
            telescopic_sum += u[m] * (sorted_distances[m + 1] - sorted_distances[m])
            model.C5.add(expr=telescopic_sum <= Dm - bound)

        telescopic_sum = 0
        for m in range(0, len(sorted_distances)-1):
            telescopic_sum += u[m] * (sorted_distances[m+1] - sorted_distances[m])
        # model.obj = pyo.Objective(expr=Dm - telescopic_sum , sense=maximize)
        model.obj = pyo.Objective(expr=1 , sense=maximize)

        print("Model Built")
        opt = SolverFactory('gurobi')
        # opt = SolverFactory('gurobi_direct')
        # opt.options['tmlim'] = 10
        opt.options['TimeLimit'] = time_max
        # If there are memory problems, then reduce the Threads
        #opt.options['Threads'] = 1
        try:
            results = opt.solve(model)
            print(results)

            if results.solver.termination_condition != 'infeasible':

                # X_value = [[i, ki] for i in range(n) for ki in range(k) if pyo.value(X[i, ki]) > 0]
                # print(X_value)
                # print([[sorted_distances[i],pyo.value(u[i])] for i in range(len(sorted_distances))])
                solution = 0
                for m in range(0, len(sorted_distances)):
                    if pyo.value(u[m]) > 0.1:
                        solution = sorted_distances[m]
                        print(solution)
                        time_value = self.extract_time_from_string(str(results["Solver"]))
                        # Guardar el diccionario en 'data.txt'
                        # self.save_dict_to_txt('output.txt', sorted_distances[m], self.instance_name, "sayah", time_value)
                        # break
                        return solution

                #print(solution)

                #print('X:', X_value)


                #return sol
                return 0
            else:
                print("Infeasible")
                return 0
        except Exception as e:
            return 0
            print(e)

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

        #model.x = pyo.Var(range(n^2), within=Binary, bounds=(0, None))

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
        print(grasp_bound, bound)

        #PR-CUT
        # model.C5 = pyo.ConstraintList()
        # telescopic_sum = 0
        #
        # for m in range(0, len(sorted_distances) - 1):
        #     telescopic_sum += u[m] * (sorted_distances[m + 1] - sorted_distances[m])
        #     model.C5.add(expr=telescopic_sum <= Dm - bound)

        # model.obj = pyo.Objective(expr=Dm - telescopic_sum , sense=maximize)
        model.obj = pyo.Objective(expr=1 , sense=maximize)

        print("Model Built")
        opt = SolverFactory('gurobi')
        # opt = SolverFactory('gurobi_direct')
        # opt.options['tmlim'] = 10
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
            print(e)