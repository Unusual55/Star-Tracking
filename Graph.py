import random

import numpy
import numpy as np

from Triangle import Edge, Vertex, Triangle


class Graph:
    def __init__(self, name, W, H):
        self.__vertices = dict()
        self.__name = name
        self.__W = W
        self.__H = H
        self.shortest_edge = None

    def add_star(self, vertex):
        for star in self.__vertices.values():
            dist = vertex.distance(star)
            radius = star.get_radius()
            if star == vertex:
                return False
            if dist <= 3 * (radius + vertex.get_radius()):
                if vertex.get_radius() > radius:
                    vertex.set_name(star.get_name())
                    self.__vertices[vertex.get_name()] = vertex
                    return True
                return False
            if self.shortest_edge is None:
                self.shortest_edge = dist
            else:
                if self.shortest_edge > dist:
                    self.shortest_edge = dist
        self.__vertices[vertex.get_name()] = vertex
        return True

    def __len__(self):
        return len(self.__vertices)

    def get_name(self):
        return self.__name

    def get_W(self):
        return self.__W

    def get_H(self):
        return self.__H

    def __iter__(self):
        return self.__vertices.values().__iter__()

    def get(self, vertex_id):
        return self.__vertices.get(vertex_id)

    def get_3_points(self):
        p1 = random.randrange(0, len(self))
        p2 = random.randrange(0, len(self))
        while p1 == p2:
            p2 = random.randrange(0, len(self))
        p3 = random.randrange(0, len(self))
        while p1 == p3 or p2 == p3:
            p3 = random.randrange(0, len(self))
        p1 = self.__vertices.get(str(p1))
        p2 = self.__vertices.get(str(p2))
        p3 = self.__vertices.get(str(p3))
        return p1, p2, p3

    def get_coordinates(self):
        X = np.array(-1)
        Y = np.array(-1)

        for star in self:
            X = numpy.append(X, star.get_x())
            Y = numpy.append(Y, star.get_y())
        X = numpy.delete(X, 0)
        Y = numpy.delete(Y, 0)
        return X, Y

    def get_triangles(self):
        T1 = []
        for i in range(len(self.__vertices) - 2):
            for j in range(i + 1, len(self.__vertices) - 1):
                for k in range(j + 1, len(self.__vertices)):
                    t = Triangle(self.__vertices.get(str(i)), self.__vertices.get(str(j)), self.__vertices.get(str(k)))
                    if t.is_relevant(self.__W, self.__H):
                        T1.append(t)
        return T1
