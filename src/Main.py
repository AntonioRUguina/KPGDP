from Utils.utils import (read_test, build_full_path)
from Utils.Instance import Instance
import numpy as np
import random
from AlgorithmByGroups import Solution_Group
from Algorithm import Solution
import time
from Gurobi import Solution_Gurobi
from PathRelinking import PathRelinking
def prepare_instance(t):
    path = build_full_path("\\Users\\Antor\\Desktop\\Git\\kpGDPAlgorithm\\KPGDP\\src\\NewInstances\\" + t.instName)
    inst = Instance(path)

    np.random.seed(t.seed)
    random.seed(t.seed)

    delta = 0.5
    prepare_dict = {"t": t, "inst": inst, "delta": delta }

    return prepare_dict





if __name__ == "__main__":

    tests = read_test("run_test.txt")
    use_ls3 = True
    verbose = True
    algorithms = ["Bias"]
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
                    sol = Solution_Group(params_dict, max_of, use_ls3)
                else:
                    sol = Solution(params_dict, max_of, use_ls3)
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
                    pr_candidates.append(final_sol.selected_dict)

            if verbose:
                print("Iterations: ", it)
                print(final_sol.of, final_sol.dict_disp_group, final_sol.n_selected, final_sol.selected_list,
                  final_sol.selected_dict)
            final_sol.save_dict_to_txt('TESTNUEVO.txt', of, t.instName, algorithm, t.max_time, t.seed)
            print((pr_candidates))
            pr = PathRelinking(params_dict,pr_candidates)
            pr.run_algorithm()

"""
if __name__ == "__main__":

    tests = read_test("run_test.txt")
    use_ls3 = True
    verbose = True
    algorithms = ["Bias"]
    for alg in algorithms:
        algorithm = alg
        for t in tests:

            print(t.instName, algorithm)
            np.random.seed(t.seed)
            params_dict = prepare_instance(t)
            pr = PathRelinking(params_dict,[{0: [80, 61, 324, 438, 98, 170, 145, 117, 273, 59, 79, 455, 429, 78, 111, 4, 149, 316, 461, 400, 77, 449, 157, 68, 494, 5, 267, 39, 120, 74, 403, 279, 1, 140, 337, 242, 199, 184, 431, 154, 312, 244, 245, 40, 284, 151, 192, 252, 132, 427], 1: [6, 238, 207, 85, 296, 366, 423, 498, 226, 46, 412, 135, 23, 8, 87, 223, 286, 218, 251, 389, 158, 378, 358, 418, 334, 487, 310, 382, 41, 385, 265, 163, 116, 367, 459, 426, 332, 187, 99, 73, 144, 405, 362, 270, 460, 90, 152, 392, 447, 50]}, {0: [28, 294, 334, 111, 330, 93, 297, 153, 159, 117, 347, 158, 308, 304, 443, 392, 17, 346, 361, 96, 439, 166, 313, 223, 342, 163, 381, 41, 260, 178, 385, 333, 6, 331, 295, 408, 139, 205, 416, 24, 241, 355, 211, 277, 351, 217, 340, 16, 177, 485], 1: [261, 135, 487, 236, 375, 55, 396, 373, 404, 120, 98, 64, 382, 233, 62, 99, 316, 140, 460, 35, 254, 267, 184, 78, 216, 227, 213, 51, 198, 148, 354, 311, 418, 459, 481, 362, 384, 179, 118, 22, 426, 7, 1, 81, 465, 271, 132, 174, 23, 328]}, {0: [29, 256, 158, 489, 134, 231, 35, 396, 82, 263, 333, 399, 81, 84, 358, 393, 310, 406, 342, 283, 70, 426, 32, 114, 1, 90, 363, 323, 445, 9, 67, 497, 30, 111, 52, 316, 267, 186, 190, 457, 274, 291, 372, 397, 51, 252, 64, 18, 260, 183], 1: [80, 222, 61, 438, 324, 98, 99, 290, 197, 451, 305, 280, 144, 79, 465, 122, 194, 413, 16, 377, 209, 407, 334, 39, 157, 117, 55, 12, 441, 173, 137, 125, 76, 230, 129, 416, 332, 132, 357, 384, 83, 460, 420, 382, 336, 484, 232, 351, 160, 210]}, {0: [22, 461, 149, 96, 106, 241, 70, 234, 487, 412, 484, 135, 380, 124, 315, 238, 288, 151, 50, 429, 316, 416, 318, 447, 274, 449, 179, 267, 198, 265, 188, 187, 245, 224, 463, 469, 117, 365, 374, 466, 52, 337, 153, 177, 432, 306, 144, 43, 439, 85], 1: [29, 373, 192, 86, 320, 262, 45, 28, 413, 391, 382, 81, 32, 462, 141, 162, 134, 231, 194, 386, 61, 19, 147, 481, 102, 173, 376, 471, 191, 269, 125, 278, 405, 332, 456, 419, 146, 5, 56, 27, 488, 492, 75, 31, 200, 74, 281, 232, 229, 101]}, {0: [6, 118, 198, 441, 451, 391, 382, 429, 364, 261, 235, 262, 323, 200, 151, 404, 316, 99, 157, 115, 460, 416, 34, 47, 449, 435, 272, 163, 339, 283, 494, 18, 280, 396, 103, 84, 354, 461, 229, 443, 363, 136, 220, 498, 390, 238, 50, 285, 177, 458], 1: [28, 294, 334, 344, 111, 30, 330, 297, 153, 159, 432, 347, 256, 426, 314, 304, 424, 403, 490, 217, 240, 313, 223, 16, 230, 281, 59, 385, 331, 284, 319, 322, 81, 488, 27, 86, 431, 439, 38, 447, 41, 165, 408, 471, 211, 21, 274, 96, 259, 402]}] )
            pr.run_algorithm()


"""
"""
pr = PathRelinking(params_dict,[{0: [110, 134, 35, 40, 25, 42, 41, 33, 143, 18, 95, 73, 91, 111, 132],
                                 1: [63, 13, 84, 9, 43, 38, 62, 107, 22, 86, 113, 8, 56, 37, 77]},
                                {0: [13, 63, 9, 21, 38, 120, 86, 62, 107, 8, 113, 37, 77, 134, 22],
                                 1: [110, 25, 42, 18, 40, 143, 33, 41, 91, 95, 73, 132, 32, 149, 35]},
                                 {0: [110, 25, 42, 40, 35, 41, 18, 143, 91, 111, 132, 32, 118, 90, 13],
                                  1: [63, 9, 38, 120, 107, 86, 22, 62, 8, 113, 37, 21, 134, 73, 102]},
                                  {0: [110, 134, 35, 40, 25, 42, 41, 33, 143, 18, 95, 73, 91, 111, 132],
                                   1: [63, 13, 84, 9, 43, 38, 62, 107, 22, 86, 113, 8, 56, 37, 77]},
                                   {0: [13, 63, 9, 21, 38, 120, 86, 62, 107, 8, 113, 37, 77, 134, 22],
                                    1: [110, 25, 42, 18, 40, 143, 33, 41, 91, 95, 73, 132, 32, 149, 35]},
                                   {0: [110, 25, 42, 40, 35, 41, 18, 143, 91, 111, 132, 32, 118, 90, 13],
                                    1: [63, 9, 38, 120, 107, 86, 22, 62, 8, 113, 37, 21, 134, 73, 102]},
                                   {0: [110, 35, 43, 111, 18, 25, 118, 40, 41, 32, 90, 143, 13, 96, 72],
                                    1: [9, 38, 21, 107, 120, 22, 62, 8, 113, 86, 56, 37, 134, 73, 42]}])
"""
