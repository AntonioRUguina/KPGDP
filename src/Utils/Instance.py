import numpy as np
from Utils.objects import Edge


class Instance:

    def __init__(self, path):
        self.name = ""  # nombre de la instancia
        self.n = 0  # nodos
        self.b = 0  # capacidad mínima requerida
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
                    self.b = int(line)
                elif i == 3:
                    a = 1
                    #line_read = line.rstrip('\t\n ')
                    #self.capacity = [float(x) for x in line_read.split('\t')]
                else:
                    line_read = line.rstrip('\t\n ')
                    d = [float(x) for x in line_read.split('\t')]
                    for z in range(0, self.n):
                        if d[z] != 0:
                            self.distance[fila, z] = d[z]
                            self.sorted_distances.append(Edge(fila, z, d[z]))
                    fila += 1
                i += 1
        self.sorted_distances.sort(key=lambda x: x.distance, reverse=True)
