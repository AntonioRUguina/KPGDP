# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:36:53 2023

@author: Antor
"""


from Main import read_test, prepare_instance
from Gurobi import Solution_Gurobi
import pandas as pd



if __name__ == "__main__":
    # Read the file with the instances to execute
    tests = read_test("run_600.txt")


    for t in tests:
        params_dict = prepare_instance(t)
        sol = Solution_Gurobi(params_dict, t.max_time)
        #sol.run_algorithm()
        sol.run_algorithm_chained()

        #print(sol.of, sol.n_selected,sol.selected_list, sol.selected_dict)

        #write_data_dinamic(sol, t, "output.txt")
    #sys.exit()



