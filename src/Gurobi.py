import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import time
import multiprocessing
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
        #Kuby
        # self.construct_solution_kgpdp(k=self.groups, p=self.p, time_max=self.max_time)
        # #Sayah
        # self.construct_solution_kgpdp_compact(k=self.groups, p=self.p, time_max=self.max_time)
        #Chained
        #self.construct_solution_kgpdp_sum(k=self.groups, p= self.p, time_max = 3600)
        #F3 Step By Step
        # bound = 0.1
        # start = time.time()
        # improved = True
        #
        # while improved:
        #     improved = False
        #     current_time = time.time() - start
        #     if (self.max_time - current_time > 1):
        #         of = self.construct_solution_kgpdp_packing(k=self.groups, p=self.p, time_max=self.max_time - current_time, bound=bound)
        #         if of > bound:
        #             improved = True
        #             bound = of
        # self.save_dict_to_txt('outputPacking.txt', bound, self.instance_name, "Chained", time.time() - start)
        # #print(self.selected_list, self.of)

        sorted_distances = list(dict.fromkeys([i.distance for i in self.sorted_distances]))
        down_index = len(sorted_distances)
        up_index = 0
        up_value = 0
        start = time.time()
        loop = True

        while loop:
            improved = False
            current_time = time.time() - start
            if (self.max_time - current_time > 1):
                target_index = int(up_index + (down_index - up_index)/2)
                if target_index == up_index:
                    loop = False
                    break
                bound = sorted_distances[target_index]
                print(up_index,down_index, target_index, bound)
                of = self.construct_solution_kgpdp_packing(k=self.groups, p=self.p, time_max=self.max_time - current_time, bound=bound)
                if of > 0:
                    bound = of
                    best_of = of
                    down_index = target_index
                else:
                    up_index = target_index


        self.save_dict_to_txt('outputPackingBinary.txt', best_of, self.instance_name, "Chained", time.time() - start)
        #print(self.selected_list, self.of)

        return self.of

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

    def construct_solution_kgpdp_compact(self, k, p, time_max):
        #pdp
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

        #model.x = pyo.Var(range(n^2), within=Binary, bounds=(0, None))

        X = model.X

        u = model.u

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
                index = sorted_distances.index(self.distance[i, j])
                for ki in range(k):
                    model.C3.add(expr=  X[i, ki] + X[j, ki] <= 1 + u[index])
        print("c3 Built")

        model.C4 = pyo.ConstraintList()
        for m in range(1, len(sorted_distances)):
            model.C4.add(expr= u[m-1] <= u[m])

        print("c4 Built")

        # model.C5 = pyo.ConstraintList()
        # for ki in range(k-1):
        #     i_sum = sum(i*X[i, ki] for i in range(n))
        #     i1_sum = sum(i*X[i, ki+1] for i in range(n))
        #     model.C5.add(expr= i_sum <= i1_sum)
        # print("c5 Built")

        telescopic_sum = 0
        for m in range(0, len(sorted_distances)-1):
            telescopic_sum += u[m] * (sorted_distances[m+1] - sorted_distances[m])
        model.obj = pyo.Objective(expr=Dm - telescopic_sum , sense=maximize)

        print("Model Built")
        opt = SolverFactory('gurobi')
        # opt = SolverFactory('gurobi_direct')
        # opt.options['tmlim'] = 10
        opt.options['TimeLimit'] = time_max
        # If there are memory problems, then reduce the Threads
        #opt.options['Threads'] = 1
        results = opt.solve(model)



        X_value = [[i, ki] for i in range(n) for ki in range(k) if pyo.value(X[i, ki]) > 0]
        print(X_value)
        #print([pyo.value(u[i]) for i in range(len(sorted_distances))])
        solution = []
        for m in range(0, len(sorted_distances)):
            if pyo.value(u[m]) > 0:
                solution.append(sorted_distances[m])
                print(solution)
                time_value = self.extract_time_from_string(str(results["Solver"]))
                # Guardar el diccionario en 'data.txt'
                self.save_dict_to_txt('output.txt', sorted_distances[m], self.instance_name, "sayah", time_value)
                break
        #print(solution)

        #print('X:', X_value)


        #return sol

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

        # model.x = pyo.Var(range(n^2), within=Binary, bounds=(0, None))

        X = model.X

        u = model.u

        model.C1 = pyo.ConstraintList()
        x_sum = sum([X[i] for i in range(n)])
        model.C1.add(expr=x_sum == p)
        #print("c1 Built")

        model.C2 = pyo.ConstraintList()

        model.C3 = pyo.ConstraintList()
        for i in range(n - 1):
            for j in range(i + 1, n):
                index = sorted_distances.index(self.distance[i][j])
                model.C3.add(expr=X[i] + X[j] <= 1 + u[index])
        #print("c3 Built")

        model.C4 = pyo.ConstraintList()
        for m in range(1, len(sorted_distances)):
            model.C4.add(expr=u[m - 1] <= u[m])

        #print("c4 Built")

        telescopic_sum = 0
        for m in range(0, len(sorted_distances) - 1):
            telescopic_sum += u[m] * (sorted_distances[m + 1] - sorted_distances[m])
        model.obj = pyo.Objective(expr=Dm - telescopic_sum, sense=maximize)

        #print("Model Built")
        opt = SolverFactory('gurobi')
        # opt = SolverFactory('gurobi_direct')
        # opt.options['tmlim'] = 10
        opt.options['TimeLimit'] = time_max
        # If there are memory problems, then reduce the Threads
        # opt.options['Threads'] = 1
        results = opt.solve(model)

        # print([pyo.value(u[i]) for i in range(len(sorted_distances))])
        solution = []
        for m in range(0, len(sorted_distances)):
            if pyo.value(u[m]) > 0:
                solution.append(sorted_distances[m])
                print(solution)
                time_value = self.extract_time_from_string(str(results["Solver"]))
                # Guardar el diccionario en 'data.txt'
                self.save_dict_to_txt('Utils/outputChained.txt', sorted_distances[m], self.instance_name, "Chained", time_value)
                break
        # print(solution)

        selected_list = [i for i in range(len(self.distance)) if pyo.value(X[i]) > 0]

        return selected_list

    def find_last_more_than_bound(self, sorted_list, bound):
        for i in range(len(sorted_list)):
            if sorted_list[i] <= bound:
                return sorted_list[i-1]
    def construct_solution_kgpdp_packing(self, k, p, time_max, bound):
        #pdp
        instance = self.instance
        n = instance.n
        model = pyo.ConcreteModel()


        sorted_distances = list(dict.fromkeys([i.distance for i in self.sorted_distances]))
        # DESCOMENTAR PARA STEP BY STEP
        # bound = self.find_last_more_than_bound(sorted_distances, bound)
        sorted_distances.reverse()
        Dm = np.max(sorted_distances)

        model.i = RangeSet(0, n)
        model.k = RangeSet(0, k)

        model.X = pyo.Var(model.i, model.k, within=Binary)

        X = model.X


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

                index = sorted_distances.index(self.distance[i, j])
                if self.distance[i, j] < bound:
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
                #print('X:', X_value)
                return solution
            else:
                print("Infeasible")
                return 0
        except Exception as e:
            return 0
            # print(e)

    """
    def construct_solution_kgpdp_sum(self, k, p,time_max):

        instance = self.instance
        n = instance.n

        model = pyo.ConcreteModel()

        model.i = RangeSet(0, n)
        model.k = RangeSet(0, k)

        model.X = pyo.Var(model.i, model.k, within=Binary)

        model.Z = pyo.Var(model.i, model.i, model.k, within=Binary)


        #model.x = pyo.Var(range(n^2), within=Binary, bounds=(0, None))

        X = model.X

        Z = model.Z
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
        model.C4 = pyo.ConstraintList()

        for i in range(n):
            for j in range(i + 1, n):
                for ki in range(k):
                    model.C3.add(expr=X[i, ki] + X[k, ki] <= 1 + Z[i, j, ki])
                    model.C4.add(expr=X[i, ki] >= Z[i, j, ki])
                    model.C4.add(expr=X[j, ki] >= Z[i, j, ki])

        model.C5 = pyo.ConstraintList()
        for i in range(n):
            for j in range(0, i):
                for ki in range(k):
                    model.C5.add(expr= 0 == Z[i, j, ki])

        sum_z = sum(self.distance[i,j] * Z[i, j, k1] for i in range(n) for j in range(i+1, n) for k1 in range(k))
        model.obj = pyo.Objective(expr=sum_z, sense=maximize)

        print("Model Built")
        opt = SolverFactory('gurobi')
        #opt = SolverFactory('gurobi_direct')
        #opt.options['tmlim'] = 10
        opt.options['TimeLimit'] = time_max
        results = opt.solve(model)

        # Función para guardar el diccionario en un archivo de texto

        #time_value = self.extract_time_from_string(str(results["Solver"]))
        # Guardar el diccionario en 'data.txt'
        #self.save_dict_to_txt('output.txt', d_value, self.instance_name,"kuby", time_value)


        #X_value = [pyo.value(X[i]) for i in range(n)]
        solution_dict = {k: [] for k in range(0, k)}
        for k in range(0, k):
            for i in range(0, n):
                if pyo.value(X[i, k]) > 0:
                    solution_dict[k].append(i)

        print (solution_dict)

    def construct_solution_pdp(self):
        # pdp
        instance = self.instance
        n = instance.n
        p = 20

        model = pyo.ConcreteModel()

        model.i = RangeSet(0, n - 1)

        model.X = pyo.Var(model.i, within= Binary)

        model.d = pyo.Var(bounds=(0, None))

        # model.x = pyo.Var(range(n^2), within=Binary, bounds=(0, None))

        X = model.X
        d = model.d
        M = np.max(self.distance)

        x_sum = sum([X[i] for i in range(n)])
        model.C1 = pyo.Constraint(expr=x_sum == p)
        model.C2 = pyo.ConstraintList()
        for i in range(n):
            for j in range(n):
                if i < j:
                    model.C2.add(expr=M * X[i] + M * X[j] + d <= 2 * M + self.distance[i, j])

        model.obj = pyo.Objective(expr=d, sense=maximize)

        opt = SolverFactory('glpk')
        # opt = SolverFactory('gurobi_direct')
        opt.options['tmlim'] = 100
        results = opt.solve(model)

        print(results)

        X_value = [pyo.value(X[i]) for i in range(n)]
        d_value = pyo.value(d)
        solution = []
        for i in range(0, n):
            if pyo.value(X[i]) > 0:
                solution.append(i)

        print(solution)

        print('X:', X_value)
        print('d:', d_value)
    """