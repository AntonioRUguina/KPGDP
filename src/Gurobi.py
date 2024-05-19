import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
class Solution_Gurobi:
    def __init__(self, param_dict):
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


        """"
        self.weight = weight
        self.max_capacity = -1
        self.max_min_dist = -1
        self.real_alpha = real_alpha
        # Dynamic enviroment
        random.seed(t.seed+count)
        """
    def run_algorithm(self):
        #self.construct_solution_kgpdp(k=self.groups, p= self.p, time_max = 3600)
        self.construct_solution_kgpdp_compact(k=self.groups, p= self.p, time_max = 3600)
        #self.construct_solution_pdp()

        #print(self.selected_list, self.of)

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

        opt = SolverFactory('gurobi')
        #opt = SolverFactory('gurobi_direct')
        #opt.options['tmlim'] = 10
        opt.options['TimeLimit'] = time_max
        results = opt.solve(model)
        d_value = pyo.value(d)


        # Función para guardar el diccionario en un archivo de texto

        time_value = self.extract_time_from_string(str(results["Solver"]))
        # Guardar el diccionario en 'data.txt'
        self.save_dict_to_txt('output.txt', d_value, self.instance_name,"kuby", time_value)


        print('d:', d_value)
        """"
        #X_value = [pyo.value(X[i]) for i in range(n)]
        solution_dict = {k: [] for k in range(0, k)}
        for k in range(0, k):
            for i in range(0, n):
                if pyo.value(X[i, k]) > 0:
                    solution_dict[k].append(i)

        print (solution_dict)

        #print('X:', X_value)
        """

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

        opt = SolverFactory('gurobi')
        # opt = SolverFactory('gurobi_direct')
        # opt.options['tmlim'] = 10
        opt.options['TimeLimit'] = time_max
        opt.options['NodeFileStart'] = 0.2
        opt.options['Threads'] = 2
        results = opt.solve(model)



        #X_value = [pyo.value(X[i]) for i in range(n)]
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

"""
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