import numpy as np
from Utils.objects import Edge


class Instance:

    def __init__(self, path):
        self.name = ""  # nombre de la instancia
        self.n = 0  # nodos
        self.p = 0
        self.k = 0
        self.capacity = []  # vector de capacidades
        self.distance = None  # matriz de distancia
        self.sorted_distances = []  # lista ordenada de Edge
        self.read_instance(path)

    def read_instance(self, s):
        with open(s) as instance:
            i = 1
            fila = 0
            for line in instance:
                if line == "\n":
                    continue
                if i == 1:
                    self.n = int(line)
                    self.distance = np.zeros((self.n, self.n))
                elif i == 2:
                    self.k = int(line)
                elif i == 3:
                    self.p = int(line)
                else:
                    line_read = line.rstrip('\t\n ')
                    d = [float(x) for x in line_read.split('\t')]
                    for z in range(0, self.n):
                        #if d[z] != 0:
                            self.distance[fila, z] = d[z]
                            self.sorted_distances.append(Edge(fila, z, d[z]))
                    fila += 1
                i += 1
        self.sorted_distances.sort(key=lambda x: x.distance, reverse=True)
        #to save in memory take only the 20% of the list
        #top_10_percent_index = int(len(self.sorted_distances) * 0.1)
        #To run Gurobi, cant reduce the matrix
        top_10_percent_index = int(len(self.sorted_distances))
        self.sorted_distances = self.sorted_distances[:top_10_percent_index]
