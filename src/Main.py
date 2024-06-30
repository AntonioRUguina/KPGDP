from Utils.utils import (read_test, build_full_path)
from Utils.Instance import Instance
import numpy as np
import random
from AlgorithmByGroups import Solution_Group
from Algorithm import Solution
import time

def prepare_instance(t):
    path = build_full_path("\\Users\\Antor\\Desktop\\Git\\kpGDPAlgorithm\\KPGDP\\src\\NewInstances\\" + t.instName)
    inst = Instance(path)

    np.random.seed(t.seed)
    random.seed(t.seed)

    delta = 0.5
    prepare_dict = {"t": t, "inst": inst, "delta": delta }

    return prepare_dict

if __name__ == "__main__":
    # Read the file with the instances to execute
    tests = read_test("run_test.txt")
    use_ls3 = False
    verbose = False
    for alg in ["Bias", "BiasByGroup"]:
        algorithm = alg
        for t in tests:
            print(t.instName)
            np.random.seed(t.seed)
            params_dict = prepare_instance(t)
            of = 0
            it = 0
            beta_0 = 0.5
            beta_1 = 0.5
            start = time.time()
            max_of = 0
            while time.time() - start < t.max_time:
                it += 1
                if algorithm == "Bias":
                    sol = Solution(params_dict, max_of, use_ls3)
                else:
                    sol = Solution_Group(params_dict, max_of, use_ls3)
                beta_0_it = np.random.triangular(0, beta_0, 1, 1)[0]
                beta_1_it = np.random.triangular(0, beta_1, 1, 1)[0]
                of_it = sol.run_algorithm(beta_0=beta_0_it, beta_1=beta_1_it)
                if of < of_it:
                    of = of_it
                    max_of = of_it
                    final_sol = sol
                    beta_0 = beta_0_it
                    beta_1 = beta_1_it
                    if verbose:
                        print(it, final_sol.historial, beta_0, beta_1, of_it)
            if verbose:
                print("Iterations: ", it)
                print(final_sol.of, final_sol.dict_disp_group, final_sol.n_selected, final_sol.selected_list,
                  final_sol.selected_dict)
            final_sol.save_dict_to_txt('outpExp2new.txt', of, t.instName, algorithm, t.max_time, t.seed)


