from Utils.utils import (read_test, build_full_path)
from Utils.Instance import Instance
import numpy as np
import random
from AlgorithmByGroups import Solution_Group
from Algorithm import Solution
import time
from GurobiFast import Solution_Gurobi
def prepare_instance(t):
    path = build_full_path("\\Users\\Antor\\Desktop\\Git\\kpGDPAlgorithm\\KPGDP\\src\\Instances\\" + t.instName)
    inst = Instance(path)

    np.random.seed(t.seed)
    random.seed(t.seed)

    delta = 0.5
    prepare_dict = {"t": t, "inst": inst, "delta": delta }

    return prepare_dict

if __name__ == "__main__":

    tests = read_test("run_60.txt")
    use_ls = True
    use_ls3 = True
    use_MRM = True
    verbose = False
    algorithms = ["Bias"]
    # algorithms = ["Bias", "BiasByGroup"]
    for alg in algorithms:
        algorithm = alg
        for t in tests:
            print(t.instName, algorithm)
            np.random.seed(t.seed)
            params_dict = prepare_instance(t)
            of = 0
            it = 0
            beta_0 = 0.5
            beta_1 = 0.5
            start = time.time()
            max_of = 0
            pr_candidates = []
            while time.time() - start < t.max_time:
                it += 1
                if algorithm == "BiasByGroup":
                    sol = Solution_Group(params_dict, max_of, use_ls, use_ls3)
                else:
                    sol = Solution(params_dict, max_of, use_ls, use_ls3)
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
                        print(final_sol.of, final_sol.dict_disp_group, final_sol.n_selected, final_sol.selected_list,
                              final_sol.selected_dict)
                if use_MRM and 0.5 * of < of_it:
                    pr_candidates.append({"solution": sol.selected_dict, "selected_list": sol.selected_list, "min_of": sol.of})


            if verbose:
                print("Iterations: ", it)
                print(final_sol.of, final_sol.dict_disp_group, final_sol.n_selected, final_sol.selected_list,
                  final_sol.selected_dict)

            final_sol.save_dict_to_txt('BIASGRASP.txt', of, t.instName, algorithm, t.max_time, t.seed)


            if use_MRM:
                sorted_solutions = sorted(pr_candidates, key=lambda x: x['min_of'], reverse=True)
                top_solutions = []
                used_nodes = set()
                m = sorted_solutions[0]["min_of"] + 1


                for sol in sorted_solutions:
                    if len(top_solutions) >= 10:
                        break

                    include = True
                    for ts in top_solutions:
                        if set(sol['selected_list']) == set(ts['selected_list']):
                            include = False
                            break
                        elif len(set(ts['selected_list'])) + 3 > len(set(ts['selected_list'] + sol['selected_list'])):
                            include = False
                            break
                    if include:
                        top_solutions.append(sol)
                        m = sol["min_of"]

                params_dict = prepare_instance(t)

                max_pr_time = 120
                start_pr = time.time()
                improved = True

                # s = 10

                while improved:
                    current_time = time.time() - start_pr
                    if (max_pr_time - current_time > 1):
                        sol = Solution_Gurobi(params_dict, round(max_pr_time-current_time),
                                              [i["selected_list"] for i in top_solutions], of)
                        of_pr = sol.run_algorithm()
                        if of_pr > of:
                            improved = True
                            print(of_pr)
                            of = of_pr
                        else:
                            improved = False
                    else:
                        break


                final_sol.save_dict_to_txt('PR10.txt', of, t.instName, "PR10Pairs", min(max_pr_time, round(time.time() - start_pr)), t.seed)
