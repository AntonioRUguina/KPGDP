import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
class Solution_Gurobi:
    def __init__(self, param_dict):
        self.instance = param_dict["inst"]
        self.delta = param_dict["delta"]
        self.sorted_distances = self.instance.sorted_distances
        self.distance = self.instance.distance
        self.selected_list = []
        self.selected_dict = {i: [] for i in range(0, param_dict["groups"])}
        self.n_selected = param_dict["groups"]*[0]
        self.v_min1 = -1
        self.v_min2 = -1
        self.of = self.sorted_distances[0].distance * 10
        self.capacity = 0
        self.time = []
        self.groups = param_dict["groups"]
        self.patron = []


        """"
        self.weight = weight
        self.max_capacity = -1
        self.max_min_dist = -1
        self.real_alpha = real_alpha
        # Dynamic enviroment
        random.seed(t.seed+count)
        """
    def run_algorithm(self):

        self.construct_solution_kgpdp(k=2)
        #self.construct_solution_pdp()

        #print(self.selected_list, self.of)

        return self.of

    def update_of(self, v_min1, v_min2, of):
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
            feasible = feasible and len(value) >= 20

        return feasible

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

        # return sol
    def construct_solution_kgpdp(self, k):

        #pdp
        instance = self.instance
        n = instance.n
        p= 20

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
        for i in range(n):
            for j in range(n):
                if i < j:
                    for ki in range(k):
                        model.C3.add(expr= M*X[i, ki] + M*X[j, ki] + d <= 2*M + self.distance[i, j] )


        model.obj = pyo.Objective(expr=d, sense=maximize)

        opt = SolverFactory('cbc')
        #opt = SolverFactory('gurobi_direct')
        opt.options['tmlim'] = 10
        results = opt.solve(model)

        print(results)

        d_value = pyo.value(d)
        print('d:', d_value)
        #X_value = [pyo.value(X[i]) for i in range(n)]
        solution_dict = {k: [] for k in range(0, k)}
        for k in range(0, k):
            for i in range(0, n):
                if pyo.value(X[i, k]) > 0:
                    solution_dict[k].append(i)

        print (solution_dict)

        #print('X:', X_value)


        #return sol

    def construct_solution_kgpdp_compact(self,k):
        #pdp
        instance = self.instance
        n = instance.n
        p = 20

        model = pyo.ConcreteModel()

        Dm = np.max(self.distance)
        print(Dm)
        sorted_distances = list(dict.fromkeys([i.distance for i in self.sorted_distances]))
        sorted_distances.reverse()
        print(sorted_distances)
        print(len(sorted_distances))
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
        for i in range(n):
            for j in range(n):
                if i < j:
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

        opt = SolverFactory('glpk')
        #opt = SolverFactory('gurobi_direct')
        opt.options['tmlim'] = 100
        results = opt.solve(model)

        print(results)

        #X_value = [pyo.value(X[i]) for i in range(n)]
        #print([pyo.value(u[i]) for i in range(len(sorted_distances))])
        solution = []
        for m in range(0, len(sorted_distances)):
            if pyo.value(u[m]) > 0:
                solution.append(sorted_distances[m])
                break

        print(solution)

        #print('X:', X_value)


        #return sol