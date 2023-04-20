from Edge import Vertex, Edge
import math


class Triangle:
    def __init__(self, p1, p2, p3):
        self.__a = p1
        self.__b = p2
        self.__c = p3

    def get_a(self):
        return self.__a

    def get_b(self):
        return self.__b

    def get_c(self):
        return self.__c

    """
    In this function we will check if two triangles are similar according to the angles
    """

    def __is_similar(self, other, eps, delta):

        ab1 = Edge(self.__a, self.__b)
        bc1 = Edge(self.__b, self.__c)
        ca1 = Edge(self.__c, self.__a)
        ab2 = Edge(other.__a, other.__b)
        bc2 = Edge(other.__b, other.__c)
        ca2 = Edge(other.__c, other.__a)

        # bac1 = 180 - math.degrees(ab1.intersection_angle(ca1))
        # abc1 = 180 - math.degrees(ab1.intersection_angle(bc1))
        # acb1 = 180 - math.degrees(ca1.intersection_angle(bc1))
        # bac2 = 180 - math.degrees(ab2.intersection_angle(ca2))
        # abc2 = 180 - math.degrees(ab2.intersection_angle(bc2))
        # acb2 = 180 - math.degrees(ca2.intersection_angle(bc2))
        #
        # angles = [bac1, abc1, acb1]
        # other_angles = [bac2, abc2, acb2]
        # check1 = sum(angles)
        # check2 = sum(other_angles)
        #
        # ang_tup = []
        # ang_tup_other = []
        # for i in range(3):
        #     for j in range(i + 1, 3):
        #         if i == j:
        #             continue
        #
        #         ang_tup.append([angles[i], angles[j]])
        #         ang_tup_other.append([other_angles[i], other_angles[j]])
        # flag = False
        #
        # for tup in ang_tup:
        #     for tup2 in ang_tup_other:
        #
        #         """
        #         If there are 2 equal angles in both triangles then the third one is equal as well and they are similar
        #         """
        #
        #         if ((abs(tup[0] - tup2[0]) < eps) and (abs(tup[1] - tup2[1]) < eps)) or (
        #                 (abs(tup[0] - tup2[1]) < eps) and (abs(tup[1] - tup2[0]) < eps)):
        #             flag = True
        #
        # if flag is not True:
        #     return False

        edges1 = [ab1.length(), bc1.length(), ca1.length()]
        edges2 = [ab2.length(), bc2.length(), ca2.length()]
        for i in range(len(edges1)):
            for j in range(len(edges1)):
                if i == j:
                    continue
                for m in range(len(edges1)):
                    e11 = edges1[i]
                    e12 = edges1[j]
                    e13 = edges1[m]
                    for k in range(len(edges2)):
                        if k == i or k == j:
                            continue
                        for l in range(len(edges2)):
                            if k == l:
                                continue
                            for n in range(len(edges2)):
                                if n == k or k == l:
                                    continue
                                e21 = edges2[i]
                                e22 = edges2[j]
                                e23 = edges2[n]
                                if ((abs(e11 / e21 - e12 / e22) < delta) or (abs(e11 / e22 - e12 / e21) < delta)) and ((
                                                                                                                               (
                                                                                                                                       abs(e11 / e13 - e12 / e23) < delta) or (
                                                                                                                                       abs(e11 / e23 - e12 / e13) < delta)) or (
                                                                                                                               (
                                                                                                                                       abs(e11 / e13 - e12 / e23) < delta) or (
                                                                                                                                       abs(e11 / e23 - e12 / e13) < delta))):
                                    return True
        return False

    def __index_to_vertex(self, index):
        if index == 0:
            return self.__a
        elif index == 1:
            return self.__b
        else:
            return self.__c

    def is_relevant(self, W, H):
        ab1 = Edge(self.__a, self.__b)
        bc1 = Edge(self.__b, self.__c)
        ca1 = Edge(self.__c, self.__a)
        bound = min(W, H) / 2
        bound_ab = min(self.__a.get_radius(), self.__b.get_radius())
        bound_ac = min(self.__a.get_radius(), self.__c.get_radius())
        bound_bc = min(self.__a.get_radius(), self.__b.get_radius())
        if ab1.length() > bound or bc1.length() > bound or ca1.length() > bound:
            return False
        if ab1.length() <= 2 * bound_ab or ca1.length() <= 2 * bound_ac or bc1.length() <= bound_bc:
            return False
        return True

    def get_similar_vertices(self, other, eps, delta):
        if not self.__is_similar(other, eps, delta):
            return None

        ab1 = Edge(self.__a, self.__b)
        bc1 = Edge(self.__b, self.__c)
        ca1 = Edge(self.__c, self.__a)

        ab2 = Edge(other.__a, other.__b)
        bc2 = Edge(other.__b, other.__c)
        ca2 = Edge(other.__c, other.__a)

        A1 = [ab1, ca1]
        A2 = [ab2, ca2]
        B1 = [ab1, bc1]
        B2 = [ab2, bc2]
        C1 = [ca1, bc1]
        C2 = [ca2, bc2]

        T1 = [A1, B1, C1]
        T2 = [A2, B2, C2]

        bac1 = 180 - math.degrees(ab1.intersection_angle(ca1))
        abc1 = 180 - math.degrees(ab1.intersection_angle(bc1))
        acb1 = 180 - math.degrees(ca1.intersection_angle(bc1))
        bac2 = 180 - math.degrees(ab2.intersection_angle(ca2))
        abc2 = 180 - math.degrees(ab2.intersection_angle(bc2))
        acb2 = 180 - math.degrees(ca2.intersection_angle(bc2))

        output = set()

        angles = [bac1, abc1, acb1]
        index = [0, 1, 2]
        index2 = [0, 1, 2]
        other_angles = [bac2, abc2, acb2]
        sum = 0
        for i in range(3):
            for j in range(3):
                # eps_confidence = abs(angles[i] - other_angles[j])
                eps_confidence = 0
                # if eps_confidence > eps:
                #     continue
                e10 = T1[i][0]
                e11 = T1[i][1]
                e20 = T2[j][0]
                e21 = T2[j][1]
                delta_confidence = 0
                d1 = abs((e10.length() / e20.length()) - (e11.length() / e21.length()))
                d2 = abs((e10.length() / e21.length()) - (e11.length() / e20.length()))
                b1 = (d1 < delta)
                b2 = (d2 < delta)
                if b1 or b2:
                    if b1:
                        delta_confidence = d1
                    if b2:
                        delta_confidence = d2
                    p1 = self.__index_to_vertex(i)
                    p2 = other.__index_to_vertex(j)
                    confidence = eps_confidence + delta_confidence
                    sum += confidence
                    output.add((p1, p2, confidence))
        return output
