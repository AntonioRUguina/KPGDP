from Utils.objects import Test
import os

def read_test(name):
    """
    Function to read de the testFile
    The testife is composed of the following parameters:
    #Instance   Seed    Time    BetaBR    BetaLs  MaxIterLS
    -Instance: Name the instance
    -Seed: Seed used to generate random numbers in the BR heuristic
    -Time: Maximum execution time
    Note: Use # to comment lines in the file
    """

    file_name = build_full_path("kpGDP\\src\\test\\" + name)
    tests = []
    with open(file_name, 'r') as tests_file:
        for line_test in tests_file:
            line_test = line_test.strip()
            if '#' not in line_test:
                line = line_test.split(';')
                test = Test(line[0], line[1], line[2])
                tests.append(test)

    return tests

def build_full_path(relative_path):
    current_path = os.getcwd()

    # Encuentra la posición de "kpGDP" en la ruta actual
    index = current_path.find("kpGDP")

    # Si no encontramos la raíz del proyecto, lanzar un error
    if index == -1:
        raise ValueError("The root" + "kpGDP" + "is not found in the current path.")

    # Construir la ruta completa hasta "kpGDP"
    base_path = current_path[:index + len("kpGDP")]

    # Retorna la ruta completa
    return os.path.join(base_path, relative_path)