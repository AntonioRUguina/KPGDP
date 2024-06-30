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

    file_name = build_full_path("KPGDP\\src\\test\\" + name)
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
    index = current_path.find("kpGDPAlgorithm")

    # Si no encontramos la raíz del proyecto, lanzar un error
    if index == -1:
        raise ValueError("The root" + "kpGDPAlgorithm" + "is not found in the current path.")

    # Construir la ruta completa hasta "kpGDP"
    base_path = current_path[:index + len("kpGDPAlgorithm")]

    # Retorna la ruta completa
    return os.path.join(base_path, relative_path)





"""
if __name__ == "__main__":

    import os

    # Define the source directory and the output file
    source_directory = "\\Users\\Antor\\Desktop\\Git\\kpGDPAlgorithm\\KPGDP\\src\\NewInstances\\"
    output_file = "../test/run_60.txt"

    # Get a list of all .txt files in the source directory
    txt_files = [f for f in os.listdir(source_directory) if f.endswith('.txt')]

    # Open the output file in write mode
    with open(output_file, 'w') as file:
        # Write the required information for each .txt file
        for txt_file in txt_files:
            file.write(f"{txt_file};7357;60\n")

    print(f"File names written to {output_file}")
"""
if __name__ == "__main__":
    import pandas as pd
    import re

    # Input data
    # Define the path to the input file
    input_file = '../outputAll.txt'

    # Read the contents of the file
    with open(input_file, 'r') as file:
        data = file.read()

    # Parse the text data
    lines = data.strip().split('\n')


    # Initialize a dictionary to store the data
    data_dict = {
        "Name": [],
        "sayah_time": [],
        "sayah_fo": [],
        "kuby_time": [],
        "kuby_fo": [],
        "Bias_time": [],
        "Bias_fo": [],
        "BiasByGroup_time": [],
        "BiasByGroup_fo": []
    }

    # Helper dictionary to temporarily store the parsed values
    temp_dict = {}

    for line in lines:
        print(line)
        parts = line.split()
        name = parts[0]
        model = parts[1]
        value1 = float(parts[2].replace(':', ''))
        value2 = float(parts[3])

        if name not in temp_dict:
            temp_dict[name] = {"sayah": [None, None], "kuby": [None, None], "Bias": [None, None], "BiasByGroup": [None, None]}

        temp_dict[name][model] = [value1, value2]

    # Transfer the data from temp_dict to data_dict
    for name, values in temp_dict.items():
        data_dict["Name"].append(name)
        data_dict["sayah_time"].append(values["sayah"][0])
        data_dict["sayah_fo"].append(values["sayah"][1])
        data_dict["kuby_time"].append(values["kuby"][0])
        data_dict["kuby_fo"].append(values["kuby"][1])
        data_dict["Bias_time"].append(values["Bias"][0])
        data_dict["Bias_fo"].append(values["Bias"][1])
        data_dict["BiasByGroup_time"].append(values["BiasByGroup"][0])
        data_dict["BiasByGroup_fo"].append(values["BiasByGroup"][1])

    # Create a DataFrame
    df = pd.DataFrame(data_dict)

    print(df)

    # Export the DataFrame to an Excel file
    output_file = "../test/outputNew.csv"
    df.to_csv(output_file, index=False, sep=';', decimal=",")

    #output_file
