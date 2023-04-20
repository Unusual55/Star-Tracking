import math

import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


from Vertex import Vertex


class Edge:
    def __init__(self, p1, p2):
        self.__p1 = p1
        self.__p2 = p2

    def get_p1(self):
        return self.__p1

    def get_p2(self):
        return self.__p2

    def length(self):
        return self.__p1.distance(self.__p2)

    def intersection_angle(self, other):
        x1 = self.__p1.get_x()
        y1 = self.__p1.get_y()
        x2 = self.__p2.get_x()
        y2 = self.__p2.get_y()
        x3 = other.__p1.get_x()
        y3 = other.__p1.get_y()
        x4 = other.__p2.get_x()
        y4 = other.__p2.get_y()
        return angle((x2 - x1, y2 - y1), (x4 - x3, y4 - y3))
        # dot_product = x1 * x2 + y1 * y2
        # modOfVector1 = math.sqrt(x1**2 + y1 ** 2) * math.sqrt(x2 ** 2 + y2 ** 2)
        # angle = dot_product/modOfVector1
        # angleInDegree = math.degrees(math.acos(angle))
        # return angleInDegree

    def __str__(self):
        output = str(self.__p1) + '\n' + str(self.p2)
        return output

    def __eq__(self, other):
        if (self.__p1 == other.__p1 and self.__p2 == other.__p2) or (
                self.__p1 == other.__p2 and self.__p2 == other.__p1):
            return True
        return False
