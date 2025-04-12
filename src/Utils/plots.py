import random
import math
import matplotlib.pyplot as plt


def generate_random_points(n, max_x=100, max_y=100):
    return [(random.uniform(0, max_x), random.uniform(0, max_y)) for _ in range(n)]


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def write_distances_to_file(points, filename="distances.txt"):
    n = len(points)
    with open(filename, "w") as file:
        for i in range(n):
            distances = []
            for j in range(n):
                distance = calculate_distance(points[i], points[j])
                distances.append(str(round(distance,2)))
            file.write("\t".join(distances) + "\n")


def plot_points(points, list1, list2,save_filename=""):
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    colors = []
    sizes = []
    for i in range(len(points)):
        if i in list1:
            colors.append('red')
            sizes.append(200)
        elif i in list2:
            colors.append('blue')
            sizes.append(200)
        else:
            colors.append('gray')
            sizes.append(100)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_values, y_values, c=colors, s=sizes, marker='o')


    plt.xlim(-1, 101)
    plt.ylim(-1, 101)

    #plt.xlabel('X coordinate')
    #plt.ylabel('Y coordinate')
    #plt.title('Random Points in 100x100 Plane')
    plt.grid(False)
    plt.savefig(save_filename)
    plt.show()

def main_map():
    n = 70
    random.seed(100)
    points = generate_random_points(n)
    #write_distances_to_file(points)
    # Define two lists of nodes for testing
    #{0: [14, 15, 19, 30, 54, 64, 68], 1: [4, 5, 21, 32, 41, 50, 67]}
    #{0: [15, 30, 50, 54, 61, 64, 68], 1: [5, 11, 20, 28, 41, 52, 67]}

    list1 = [14, 15, 19, 30, 54, 64, 68]# First list of nodes
    list2 = [4, 5, 21, 32, 41, 50, 67] # Second list of nodes

    plot_points(points, list1, list2, "sum_plot")


def main_boxplo_2_models():
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
            temp_dict[name] = {"sayah": [None, None], "kuby": [None, None], "Bias": [None, None],
                               "BiasByGroup": [None, None]}

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

    # Transcribed data

    df['max_gurobi'] = df[['sayah_fo', 'kuby_fo']].max(axis=1)
    df['SD_GAP'] = (df['max_gurobi'] - df['Bias_fo'])/df['max_gurobi'] * 100
    df['GD_GAP'] = (df['max_gurobi'] - df['BiasByGroup_fo'])/df['max_gurobi'] * 100

    # Split the data into four lists
    GKD_50S = df[df['Name'].str.startswith('GKD-b') & df['Name'].str.contains('_n50_')]["SD_GAP"]
    GKD_50G = df[df['Name'].str.startswith('GKD-b') & df['Name'].str.contains('_n50_')]["GD_GAP"]
    GKD_150S = df[df['Name'].str.startswith('GKD-b') & df['Name'].str.contains('_n150_')]["SD_GAP"]
    GKD_150G = df[df['Name'].str.startswith('GKD-b') & df['Name'].str.contains('_n150_')]["GD_GAP"]
    GKD_500S = df[df['Name'].str.startswith('GKD-c')]["SD_GAP"]
    GKD_500G = df[df['Name'].str.startswith('GKD-c')]["GD_GAP"]
    MDG_500S = df[df['Name'].str.startswith('MDG')]["SD_GAP"]
    MDG_500G = df[df['Name'].str.startswith('MDG')]["GD_GAP"]
    RUMG_500S = df[df['Name'].str.startswith('RUMG') & df['Name'].str.contains('_n500_')]["SD_GAP"]
    RUMG_500G = df[df['Name'].str.startswith('RUMG') & df['Name'].str.contains('_n500_')]["GD_GAP"]

    # Combine the data for plotting
    data_50 = [GKD_50S, GKD_50G]
    data_150 = [GKD_150S, GKD_150G]
    data_gkd500 = [GKD_500S, GKD_500G]
    data_mdg500 = [MDG_500S, MDG_500G]
    data_rumg500 = [RUMG_500S, RUMG_500G]

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

    colors = ['lightblue', 'lightgreen']

    # Plot the boxplots for set data_50
    #bp_A = axes[0].boxplot(data_50, patch_artist=True)
    #for patch, color in zip(bp_A['boxes'], colors):
    #    patch.set_facecolor(color)

    #axes[0].set_title('Set GKD n50')
    #axes[0].set_xticklabels(['SD', 'GD'])

    # Plot the boxplots for set data_150
    bp_B = axes[0][0].boxplot(data_150, patch_artist=True)
    for patch, color in zip(bp_B['boxes'], colors):
        patch.set_facecolor(color)
    axes[0][0].set_title('Set GKD n150')
    axes[0][0].set_xticklabels(['SD', 'GD'])
    axes[0][0].set_ylabel("% Deviation")

    # Plot the boxplots for set data_gkd500
    bp_C = axes[0][1].boxplot(data_gkd500, patch_artist=True)
    for patch, color in zip(bp_C['boxes'], colors):
        patch.set_facecolor(color)
    axes[0][1].set_title('Set GKD n500')
    axes[0][1].set_xticklabels(['SD', 'GD'])
    axes[0][1].set_ylabel("% Deviation")

    # Plot the boxplots for set data_mdg500
    bp_D = axes[1][0].boxplot(data_mdg500, patch_artist=True)
    for patch, color in zip(bp_D['boxes'], colors):
        patch.set_facecolor(color)
    axes[1][0].set_title('Set MDG n500')
    axes[1][0].set_xticklabels(['SD', 'GD'])
    axes[1][0].set_ylabel("% Deviation")
    # Display the plot

    # Plot the boxplots for set data_rumg500
    bp_E = axes[1][1].boxplot(data_rumg500, patch_artist=True)
    for patch, color in zip(bp_E['boxes'], colors):
        patch.set_facecolor(color)
    axes[1][1].set_title('Set RUMG n500')
    axes[1][1].set_xticklabels(['SD', 'GD'])
    axes[1][1].set_ylabel("% Deviation")
    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_map()
    #main_boxplo_2_models()

