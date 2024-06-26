from Utils.utils import (read_test,build_full_path)
from Utils.Instance import Instance
import numpy as np
import random
from Algorithm import Solution
import time

def prepare_instance(t):
    path = build_full_path("kpGDP\\src\\Instances\\" + t.instName)
    inst = Instance(path)
    n_types = 5

    np.random.seed(t.seed)
    random.seed(t.seed)

    groups = 4

    delta = 0.5
    prepare_dict = {"t": t, "inst": inst, "delta": delta, "groups": groups }

    return prepare_dict




if __name__ == "__main__":
    # Read the file with the instances to execute
    tests = read_test("run_test.txt")



    for t in tests:
        params_dict = prepare_instance(t)
        p=20
        sol = Solution(params_dict, p)
        #sol.analizar_patron([5, 6, 12, 16, 18, 21, 22, 24, 25, 26, 27, 28, 33, 36, 37, 40, 42, 44, 45, 49] )
        #sol.analizar_patron({0: [76, 26, 91, 19, 111, 126, 137, 14, 24, 79, 86, 125, 62, 45, 64, 31, 12, 119, 60, 0], 1: [105, 68, 22, 75, 147, 81, 117, 141, 4, 2, 34, 145, 89, 50, 21, 106, 139, 42, 114, 96]})
        #sol.analizar_patron({0: [97, 117, 52, 62, 14, 111, 76, 41, 79, 81, 0, 45, 66, 12, 61, 126, 24, 105, 125, 78], 1: [91, 141, 137, 26, 44, 80, 68, 73, 147, 145, 17, 119, 93, 50, 60, 64, 34, 13, 106, 86]})
        of = 0
        it = 0
        beta_0 = 0.5
        beta_1 = 0.5
        start = time.time()
        while time.time() - start < 600:
            it += 1
            sol = Solution(params_dict, p)
            beta_0_it = np.random.triangular(0, beta_0, 1, 1)[0]
            beta_1_it = np.random.triangular(0, beta_1, 1, 1)[0]
            of_it = sol.run_algorithm(beta_0=beta_0_it, beta_1=beta_1_it)
            if of < of_it:
                of = of_it
                final_sol = sol
                beta_0 = beta_0_it
                beta_1 = beta_1_it
                print(it,final_sol.historial, final_sol.improved_ls_group, final_sol.n_ls, beta_0, beta_1, of_it,)
        print("Iterations: ", it)
        #print(final_sol.of, final_sol.dict_disp_group, final_sol.n_selected, final_sol.selected_list,
        #      final_sol.selected_dict)
        #write_data_dinamic(sol, t, "output.txt")
    #sys.exit()
