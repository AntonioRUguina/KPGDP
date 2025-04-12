# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:36:53 2023

@author: Antor
"""


from Main import read_test, prepare_instance
from MathModels import Solution_Gurobi


if __name__ == "__main__":
    # Read the file with the instances to execute
    tests = read_test("run_60.txt")


    for t in tests:
        params_dict = prepare_instance(t)
        sol = Solution_Gurobi(params_dict, t.max_time)
        sol.run_algorithm()
        sol.run_algorithm_chained()




