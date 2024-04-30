class Test:
    def __init__(self, inst_name, seed, time):
        self.instName = inst_name
        self.seed = int(seed)
        self.max_time = int(time)

class Edge:
    def __init__(self, v1, v2, distance):
        self.v1 = v1
        self.v2 = v2
        self.distance = distance

class Node:
    def __int__(self, id):
        self.id = id

class Candidate:

    def __init__(self, v, closest_v, cost, group):
        self.v = v
        self.closest_v = closest_v
        self.cost = cost
        self.group = group

