import copy
import os
import random
from Assignment import Assignment
from Triangle import Triangle
from star_detect import detect_stars2
from Graph import Graph
from writer import write_assignments
import cv2

GR = 1.61803398875


class Optimizer:
    def __init__(self, g1, g2, g1_tri=None, g2_tri=None):
        self.matches = dict()
        self.__g1 = g1
        self.__g2 = g2
        self.__g1_tri = g1_tri
        self.__g2_tri = g2_tri
        # if g1_tri is None or g2_tri is None:
        #     self.set_triangles()
        self.__eps = 1
        self.__delta = random.uniform(0, 0.25)
        # self.__delta = 0.05
        self.fitness = 0
        self.assignments = dict()

    def set_triangles(self):
        self.__g1_tri = self.__g1.get_triangles()
        self.__g2_tri = self.__g2.get_triangles()

    def get_g1(self):
        return copy.deepcopy(self.__g1)

    def get_g1_tri(self):
        return self.__g1_tri

    def get_g2(self):
        return copy.deepcopy(self.__g2)

    def get_g2_tri(self):
        return self.__g2_tri

    def get_eps(self):
        return self.__eps

    def get_delta(self):
        return self.__delta

    def set_eps(self, eps):
        self.__eps = eps

    def set_delta(self, delta):
        self.__delta = delta

    def get_copy(self):
        return copy.deepcopy(self)

    def SwapCrossover(self, parentB):
        childA = Optimizer(self.__g1, self.__g2, self.__g1_tri, self.__g2_tri)
        childB = Optimizer(self.__g1, self.__g2, self.__g1_tri, self.__g2_tri)
        choice = random.randrange(0, 10)
        if choice > 4:
            childA.set_eps(self.__eps)
            childA.set_eps(parentB.__delta)
            childB.set_delta(self.__delta)
            childB.set_eps(parentB.__eps)
        else:
            childA.set_eps(parentB.__eps)
            childA.set_eps(self.__delta)
            childB.set_delta(parentB.__delta)
            childB.set_eps(self.__eps)

        # The children might learn some possible assignments from their parents

        for assignment in self.assignments:
            g1_id = assignment.g1_id
            g2_id = assignment.g2_id
            confidence = assignment.con
            in_data = {g2_id: [confidence]}
            if childB.matches.get(g1_id) is not None and childB.matches.get(g1_id).get(g2_id) is not None:
                childB.matches[g1_id][g2_id].append(confidence)
            else:
                childB.matches[g1_id] = in_data

            # childA.matches[g1_id] = in_data
            if childA.matches.get(g1_id) is not None and childA.matches.get(g1_id).get(g2_id) is not None:
                childA.matches[g1_id][g2_id].append(confidence)
            else:
                childA.matches[g1_id] = in_data

        for assignment in parentB.assignments:
            g1_id = assignment.g1_id
            g2_id = assignment.g2_id
            confidence = assignment.con
            in_data = {g2_id: [confidence]}

            if childB.matches.get(g1_id) is not None and childB.matches.get(g1_id).get(g2_id) is not None:
                childB.matches[g1_id][g2_id].append(confidence)
            else:
                childB.matches[g1_id] = in_data

            # childA.matches[g1_id] = in_data
            if childA.matches.get(g1_id) is not None and childA.matches.get(g1_id).get(g2_id) is not None:
                childA.matches[g1_id][g2_id].append(confidence)
            else:
                childA.matches[g1_id] = in_data

        return [childA, childB]

    def avg_mutata(self, parentB):
        childA, childB = self.SwapCrossover(parentB)
        choice = random.randrange(0, 10)
        if choice > 4:
            mutated = childA.get_copy()
            mutated.set_eps((childA.__eps + childB.__eps) / 2)
        else:
            mutated = childA.get_copy()
            mutated.set_eps((childA.__delta + childB.__delta) / 2)
        return [childA, childB, mutated]

    def inc_dec_mutate(self, parentB):
        childA, childB = self.SwapCrossover(parentB)
        choice = random.randrange(0, 10)
        rate = 0.1
        if choice > 4:
            mutated_a = childA.get_copy()
            mutated_a.set_eps(childA.__eps * (1 + rate))
            mutated_a2 = childB.get_copy()
            mutated_a2.set_eps(childB.__eps * (1 + rate))
            mutated_b = childA.get_copy()
            mutated_b.set_eps(childA.__eps * (1 - rate))
            mutated_b2 = childB.get_copy()
            mutated_b2.set_eps(childB.__eps * (1 - rate))

        else:
            mutated_a = childA.get_copy()
            mutated_a.set_delta(childA.__delta * (1 + rate))
            mutated_a2 = childB.get_copy()
            mutated_a2.set_delta(childB.__delta * (1 + rate))
            mutated_b = childA.get_copy()
            mutated_b.set_delta(childA.__delta * (1 - rate))
            mutated_b2 = childB.get_copy()
            mutated_b2.set_delta(childB.__delta * (1 - rate))

        return [childA, childB, mutated_b, mutated_b2, mutated_a, mutated_a2]

    def calculate_fitness(self, attempts):
        self.fitness = 0
        # for t1 in self.__g1_tri:
        #     for t2 in self.__g2_tri:
        for attempt in range(attempts):
            t1_a, t1_b, t1_c = self.__g1.get_3_points()
            t2_a, t2_b, t2_c = self.__g2.get_3_points()
            t1 = Triangle(t1_a, t1_b, t1_c)
            if len(self.__g1) > 2 and self.__g1.shortest_edge > min(self.__g1.get_H(), self.__g1.get_W()):
                while not t1.is_relevant(W=self.__g1.get_W(), H=self.__g1.get_H()):
                    t1_a, t1_b, t1_c = self.__g1.get_3_points()
                    t1 = Triangle(t1_a, t1_b, t1_c)
            t2 = Triangle(t2_a, t2_b, t2_c)
            if len(self.__g2) > 2 and self.__g2.shortest_edge > min(self.__g2.get_H(), self.__g2.get_W()):
                while not t2.is_relevant(W=self.__g2.get_W(), H=self.__g2.get_H()):
                    t2_a, t2_b, t2_c = self.__g2.get_3_points()
                    t2 = Triangle(t2_a, t2_b, t2_c)
            output = t1.get_similar_vertices(t2, eps=self.__eps, delta=self.__delta)
            if output is not None:
                # self.fitness += 1
                for out in output:
                    # print(f'{out[0].get_name()}, {out[1].get_name()}')
                    if self.matches.get(out[0].get_name()) is None:
                        self.matches[out[0].get_name()] = dict()
                    if self.matches[out[0].get_name()].get(out[1].get_name()) is None:
                        self.matches[out[0].get_name()][out[1].get_name()] = []
                    self.matches[out[0].get_name()][out[1].get_name()].append(out[2])
                    # self.fitness += 1
        if (len(self.matches)) > 0:
            self.assignments = self.get_assignments()
        return self.fitness

    def get_assignments(self):
        # fix the matches so it will contain only the best confidence of each match
        matches = copy.deepcopy(self.matches)
        for g1_id, g2_v in self.matches.items():
            for g2_id, cc in g2_v.items():
                k = min(cc)
                matches[g1_id][g2_id] = None
                matches[g1_id][g2_id] = k
                # = min(cc[1:-1])
        stack = []
        result = [x for x in range(len(self.__g2))]

        # find the most probable matches

        for g1_id, g2_v in self.matches.items():
            taken_a = None
            taken_g2_id = -1
            for g2_id, cc in g2_v.items():
                a = Assignment(g1_id, g2_id, min(cc))
                if taken_a is None:
                    taken_a = a
                    taken_g2_id = g2_id
                else:
                    if a < taken_a:
                        taken_a = a
                        taken_g2_id = g2_id
            item = result[int(taken_g2_id)]
            if isinstance(item, int):
                result[int(taken_g2_id)] = taken_a
                matches[g1_id].pop(taken_g2_id)
            else:
                if taken_a < result[int(taken_g2_id)]:
                    stack.append(result[int(taken_g2_id)].g1_id)
                    result[int(taken_g2_id)] = taken_a
                    matches[g1_id].pop(taken_g2_id)

        # find the second level matches

        # calculate the fitness

        final_result = []
        for item in result:
            if isinstance(item, Assignment):
                if item.con > self.__delta:
                    # self.fitness += (2 * (self.__eps + self.__delta)) - item.con
                    self.fitness += (2 * self. __eps + self.__delta) - item.con
                else:
                    # self.fitness += (self.__eps + self.__delta) - item.con
                    self.fitness += self. __eps + self.__delta - item.con
                final_result.append(item)
        return final_result

    def display_matches(self):
        img1 = cv2.imread(self.__g1.get_name())
        for star in self.__g1:
            cv2.circle(img1, (star.get_x(), star.get_y()), star.get_radius(), (0, 0, 255), 2)
            text = star.get_name()
            coordinates = (star.get_x(), star.get_y() + 2 * star.get_radius())
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 255)
            thickness = 2
            cv2.putText(img1, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        img2 = cv2.imread(self.__g2.get_name())

        for star in self.__g2:
            cv2.circle(img2, (star.get_x(), star.get_y()), star.get_radius(), (0, 0, 255), 2)
            text = star.get_name()
            coordinates = (star.get_x(), star.get_y() + 2 * star.get_radius())
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 255)
            thickness = 2
            cv2.putText(img2, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.namedWindow("g1 stars", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        cv2.namedWindow("g2 stars", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        cv2.imshow("g1 stars", img1)
        cv2.imshow("g2 stars", img2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def generate(img1, img2):
    g1 = read_graph(img1)
    if len(g1) == 0:
        g1 = detect_stars2(img1)
    print("Loaded G1!")
    g2 = read_graph(img2)
    if len(g2) == 0:
        g2 = detect_stars2(img2)
    print("Loaded G2!")
    s_1 = Optimizer(g1, g2)
    s_1.calculate_fitness(1000)
    population_bound = 50
    generation_bound = 50
    population = [(s_1, s_1.fitness)]
    print("Creating the first population!")
    for i in range(population_bound):
        sample = Optimizer(g1, g2, s_1.get_g1_tri(), s_1.get_g2_tri())
        fitness = sample.calculate_fitness(750)
        population.append((sample, fitness))

    for gen in range(generation_bound):
        population = sorted(population, key=lambda x: x[1], reverse=True)
        parentA = population[0][0]
        parent_selection = random.randrange(1, 101)
        parentB = None
        if parent_selection <= 90:
            parentB = population[1][0]
        else:
            parent_selection = random.randrange(0, len(population) - 1)
            parentB = population[parent_selection][0]
        children_selection = random.randrange(1, 101)
        children = []
        if children_selection <= 90:
            children = parentA.SwapCrossover(parentB)
            for x in children:
                x.calculate_fitness(500)
                population.append((x, x.fitness))
        else:
            mutation_selection = random.randrange(1, 101)
            if mutation_selection < 75:
                children = parentA.inc_dec_mutate(parentB)
                for x in children:
                    x.calculate_fitness(500)
                    population.append((x, x.fitness))
            else:
                children = parentA.avg_mutata(parentB)
                for x in children:
                    x.calculate_fitness(500)
                    population.append((x, x.fitness))
        if gen % 50 == 0:
            print(f'Generation {gen}:\tPopulation size: {len(population)}\tBest fitness: {parentA.fitness}')
    population = sorted(population, key=lambda x: x[1], reverse=True)
    best_opt = population[0][0]
    print(f'Generation {generation_bound}:\tPopulation size: {len(population)}\tBest fitness: {best_opt.fitness}')
    name1 = best_opt.get_g1().get_name()
    name2 = best_opt.get_g2().get_name()
    best_assignments = best_opt.assignments
    write_assignments(best_assignments, name1, name2)
    for assignment in population[0][0].assignments:
        print(assignment)
    best_opt.display_matches()


def read_graph(image_name):
    import cv2
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    W, H = gray.shape[::-1]

    parts = image_name.split('/')
    g = Graph(name=image_name, W=W, H=H)
    name = parts[-1][:-3] + 'txt'
    path = 'star_data/'
    entries = os.listdir(path)
    file_path = ''
    if name in entries:
        file_path = path + name
        with open(file_path, "r+") as file1:
            # Reading from a file
            data = file1.read()
            rows = data.split('\n')
            for i in range(1, len(rows)):
                prop = rows[i].split(', ')
                idx = prop[0]
                x = int(prop[1])
                y = int(prop[2])
                r = int(prop[3])
                b = float(prop[4])
                from Vertex import Vertex
                v = Vertex(idx, x, y, r, b)
                g.add_star(v)
    return g


generate('Star_Images/Formatted/IMG_3053.jpg', 'Star_Images/Formatted/IMG_3054.jpg')
# read_graph('Star_Images/Formatted/IMG_3053.jpg')
