from gurobipy import *
#import instance_mmdp
import instance2_gmdp
from bisect import bisect_left



def evalSolution(inst, sol):
    minDist = 0x3f3f3f
    n = len(sol)
    for i in range(n):
        ii, it = sol[i] # Extract the first pair (i, t1)
        for j in range(i+1, n):
            jj, jt = sol[j]  # Extract the second pair (j, t2)
            if it == jt:
                minDist = min(minDist, inst['d'][ii][jj])
    return minDist


try:
    #path = "instances/mdplib/GKD_d_1_n25_coor.txt"
    #p = 4
    #k = 3
    path = "instances/mdplib/MDG-b_01_n500_b02_m50.txt"
    # path = "instances/mdplib/GKD-b_41_n150_b03_m15.txt"
    p = 15
    k = 10
    inst = instance2_gmdp.readInstance(path, p, k)
   # inst = instance_mmdp.readInstance(path, p, k)
    n = inst['n']
    p = inst['p']
    k = inst['k']
    dmax = inst['distance'][len(inst['distance'])-1]
    dmin = inst['distance'][0]
    timeLimit = 600
    secs = 0
    bestSol = None
    while dmax - dmin > 0.01 and secs < timeLimit:
        l = round(dmin + (dmax - dmin) / 2, 2)

        model = Model("MMDP")
        model.setParam("LogToConsole", 0)
        model.setParam("TimeLimit", 300)

        # VARIABLES
        x = model.addVars(n, k, vtype=GRB.BINARY, name="x")

        # OBJECTIVE FUNCTION
        model.setObjective(quicksum(x[i, t] for i in range(n) for t in range(k)), GRB.MAXIMIZE)

        # CONSTRAINTS
        # R1: \sum_{i=0}^{n} x[i,t] == p  1 \leq t \leq k
        for t in range(k):
            model.addConstr((quicksum(x[i, t] for i in range(n)) == p))

        # R2: \sum_{t=0}^{k-1} x[i,t] == p  1 \leq i \leq n
        for i in range(n):
            model.addConstr(quicksum(x[i, t] for t in range(k)) <= 1, name=f"R2_{t}")

        # R3: x_it + x_jt \leq 1 if d_{ij} \geq l
        for t in range(k):
            for i in range(n):
                for j in range(i+1, n):
                    if round(inst['d'][i][j], 2) < l:
                        model.addConstr(x[i, t] + x[j, t] <= 1)

        # SOLVE THE MODEL
        model.optimize()

        status = model.getAttr("Status")
        if status == GRB.OPTIMAL:
            #print("SOL FOUND")
            sol = []
            for v in model.getVars():
                if v.x > 0 and "x" in v.varName:
                    id = int(v.varName[v.varName.find("[")+1:v.varName.find(",")])
                    idt = int(v.varName[v.varName.find(",") + 1:v.varName.find("]")])
                    sol.append((id,idt))
            dmin = round(evalSolution(inst, sol), 2)
            bestSol = sol[:]
        elif status != GRB.TIME_LIMIT:
            print("Solution not found with dmin="+str(dmin)+" and dmax="+str(dmax))
            dmax = round(inst['distance'][bisect_left(inst['distance'], l)-1], 2)
        else:
            print("Solution not found with code: "+str(status))
            break

    print("Best solution: "+str(round(evalSolution(inst, bestSol), 2)))

except GurobiError as e:
    print(e)