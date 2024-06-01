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
    for i in range(len(points)):
        if i in list1:
            colors.append('red')
        elif i in list2:
            colors.append('blue')
        else:
            colors.append('gray')

    plt.figure(figsize=(8, 8))
    plt.scatter(x_values, y_values, c=colors, s=100, marker='o')


    plt.xlim(-1, 101)
    plt.ylim(-1, 101)

    #plt.xlabel('X coordinate')
    #plt.ylabel('Y coordinate')
    #plt.title('Random Points in 100x100 Plane')
    plt.grid(False)
    plt.savefig(save_filename)
    plt.show()

def main():
    n = 70
    random.seed(100)
    points = generate_random_points(n)
    #write_distances_to_file(points)
    # Define two lists of nodes for testing
    #{0: [14, 15, 19, 30, 54, 64, 68], 1: [4, 5, 21, 32, 41, 50, 67]}
    #{0: [15, 30, 50, 54, 61, 64, 68], 1: [5, 11, 20, 28, 41, 52, 67]}

    list1 = [15, 30, 50, 54, 61, 64, 68]# First list of nodes
    list2 = [5, 11, 20, 28, 41, 52, 67]  # Second list of nodes

    plot_points(points, list1, list2, "sum_plot")


if __name__ == "__main__":
    main()
