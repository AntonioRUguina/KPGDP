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
    source_directory = "\\Users\\Antor\\Desktop\\Git\\kpGDPAlgorithm\\KPGDP\\src\\Instances\\"
    output_file = "../test/run_60.txt"

    # Get a list of all .txt files in the source directory
    txt_files = [f for f in os.listdir(source_directory) if f.endswith('.txt')]

    # Open the output file in write mode
    with open(output_file, 'w') as file:
        # Write the required information for each .txt file
        for txt_file in txt_files:
            file.write(f"{txt_file};7357;60\n")

    print(f"File names written to {output_file}")



if __name__ == "__main__":
    import pandas as pd
    import re

    # Input data
    # Define the path to the input file
    input_file = '../outputPackingSbS.txt'

    # Read the contents of the file
    with open(input_file, 'r') as file:
        data = file.read()

    # Parse the text data
    lines = data.strip().split('\n')


    # Initialize a dictionary to store the data
    data_dict = {
        "Name": [],
        "F3_time": [],
        "F3_fo": [],

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
            temp_dict[name] = {"Packing": [None, None]}

        temp_dict[name][model] = [value1, value2]

    # Transfer the data from temp_dict to data_dict
    for name, values in temp_dict.items():
        data_dict["Name"].append(name)
        data_dict["F3_time"].append(values["Packing"][0])
        data_dict["F3_fo"].append(values["Packing"][1])


    # Create a DataFrame
    df = pd.DataFrame(data_dict)

    print(df)

    # Export the DataFrame to an Excel file
    output_file = "../test/outputPackingTable.csv"
    df.to_csv(output_file, index=False, sep=';', decimal=",")

    #output_file
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.array([6, 9, 10, 15, 20, 22, 30, 50, 75])

    sayah = np.array([0, 0, 24.92, 12.50, 19.50,0, 16.77, 0, 1.69])
    chained = np.array([1.74, 3.21 ,0, 0.30, 2.14, 1.85, 0, 2.85, 3.75])

    plt.plot(x,sayah,label='Sayah',color='red', marker ='o')
    plt.plot(x,chained,label='Chained',color='blue', marker= 'o')
    plt.xlabel('p')
    plt.ylabel('mean dev')
    plt.legend()
    plt.show()

    x = np.array([2,5,10])

    sayah = np.array([0.63, 13.60, 20.79])
    chained = np.array([2.85, 1.42, 0])

    plt.plot(x, sayah, label='Sayah', color='red', marker='o')
    plt.plot(x, chained, label='Chained', color='blue', marker='o')
    plt.xlabel('k')
    plt.ylabel('mean dev')
    plt.legend()
    plt.show()
"""
if __name__ == "__main__":
    import csv
    from collections import defaultdict

    # Initialize dictionaries to store sums and minimums
    sum_dict = defaultdict(float)
    min_dict = defaultdict(lambda: float('inf'))

    # Read the data from the txt file
    with open('outputChained.txt', 'r') as file:
        for line in file:
            parts = line.split()
            name = parts[0]
            third_column = float(parts[2].rstrip(':'))
            fourth_column = float(parts[3])

            # Sum the third column values
            sum_dict[name] += third_column

            # Update the minimum value for the fourth column
            if fourth_column < min_dict[name]:
                min_dict[name] = fourth_column

    # Write the results to a CSV file
    with open('outputChained.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Name', 'Sum', 'Min'])

        for name in sum_dict:
            writer.writerow([name, sum_dict[name], min_dict[name]])
            
"""

