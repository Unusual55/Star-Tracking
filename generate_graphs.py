import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from Assignment import Assignment
from Triangle import Triangle
from Graph import Graph
from Vertex import Vertex
from writer import write_assignments
import cv2
from star_detect import detect_stars2
from scipy.spatial import KDTree


def getxy(point):
    if len(point) == 2:
        x, y = point
        return x, y
    if len(point) == 4:
        x, y, _, _ = point
        return x, y
    return None


def find_k_nearest_neighbors(points, query_point, K=1):
    # convert the list of points to a numpy array
    x_, y_, _, _ = query_point
    for i in range(len(points)):
        x, y, r, b = points[i]
        if x == x_ and y == y_:
            points[i] = (1000000000, 1000000000, r, b)
            break
    points = [(x, y) for x, y, _, _ in points]
    query_point = (x_, y_)
    points_array = np.array(points)

    # create a KDTree object with the points
    kdtree = KDTree(points_array[:, :2])  # using only the x and y coordinates

    # find the indices of the K nearest neighbors of the query point
    distances, indices = kdtree.query([query_point], k=K)

    # retrieve the corresponding points from the original array
    nearest_neighbors = [points[i] for i in indices[0]]

    return distances, nearest_neighbors


def dist_points(xa, ya, xb, yb):
    dx = xb - xa
    dy = yb - ya
    return math.sqrt(dx ** 2 + dy ** 2)


def get_inliers(g, a, b, dist_thresh=5):
    inliers = []
    xa, ya = getxy(a)
    xb, yb = getxy(b)
    m = (yb - ya) / (xb - xa)
    c = ya - m * xa

    for v in g:
        x, y, _, _ = v
        y_t_up = m * x + c + dist_thresh
        y_t_down = m * x + c - dist_thresh

        if y_t_down <= y <= y_t_up:
            inliers.append(v)

    return inliers, m, c


def same_point(a, b):
    xa, ya = getxy(a)
    xb, yb = getxy(b)
    return xa != xb and ya != yb


def valid_triangles(points1, points2, threshold, validation_threshold):
    if len(points1) != 3 or len(points2) != 3:
        raise ValueError("can only compare 3 points")

    a1, b1, c1 = points1
    a2, b2, c2 = points2

    if not (same_point(a1, b1) and same_point(a1, c1) and same_point(b1, c1)):
        return False
    if not (same_point(a2, b2) and same_point(a2, c2) and same_point(b2, c2)):
        return False

    x1, y1 = getxy(a1)
    x2, y2 = getxy(b1)
    x3, y3 = getxy(c1)

    t1 = Triangle(Vertex(name="a1", x=x1, y=y1, radius=0, brightness=0),
                  Vertex(name="b1", x=x2, y=y2, radius=0, brightness=0),
                  Vertex(name="c1", x=x3, y=y3, radius=0, brightness=0))

    m1 = (y2 - y1) / (x2 - x1)
    m2 = (y3 - y2) / (x3 - x2)
    m3 = (y3 - y1) / (x3 - x1)

    if not (m1 != m2 != m3):
        return False

    x1, y1 = getxy(a2)
    x2, y2 = getxy(b2)
    x3, y3 = getxy(c2)

    m1 = (y2 - y1) / (x2 - x1)
    m2 = (y3 - y2) / (x3 - x2)
    m3 = (y3 - y1) / (x3 - x1)

    if not (m1 != m2 != m3):
        return False

    t2 = Triangle(Vertex(name="a2", x=x1, y=y1, radius=0, brightness=0),
                  Vertex(name="b2", x=x2, y=y2, radius=0, brightness=0),
                  Vertex(name="c2", x=x3, y=y3, radius=0, brightness=0))

    return t1.is_similar(other=t2, eps=threshold, delta=validation_threshold)


def calc_affine_transformation_matrix(src, dest):
    '''
    courtesy of @Raz Gavrieli, with our own tweak on it (supports more than just 3 points)
    '''
    # dropping extra values
    src = [(x, y) for x, y, *_ in src]
    dest = [(x, y) for x, y, *_ in dest]

    n = len(src)
    if n < 3:
        raise ValueError("At least three points are required.")

    # Construct the matrix A
    A = np.zeros((2 * n, 6))
    for i in range(n):
        x, y = src[i]
        u, v = dest[i]
        A[2 * i, :] = [x, y, 1, 0, 0, 0]
        A[2 * i + 1, :] = [0, 0, 0, x, y, 1]

    # Compute the vector b
    b = np.array(dest).flatten()

    # Solve the linear system Ax=b using SVD
    U, S, Vt = np.linalg.svd(A)
    V = Vt.T
    Sinv = np.zeros((6, 2 * n))
    Sinv[:6, :6] = np.diag(1 / S[:6])
    T = V.dot(Sinv).dot(U.T).dot(b)
    T = np.concatenate((T, [0, 0, 1]))
    T = T.reshape((3, 3))

    return T


def ransac_line_fit(graph, threshold=5, max_iterations=1000, inlier_thresh=0.5):
    best_fit = None
    best_count = 0
    points_on_line = []
    for _ in range(max_iterations):
        # Randomly select two points from the set
        sample = random.sample(graph, 2)
        # Fit a line to the selected points
        inliers, m, b = get_inliers(g=graph, a=sample[0], b=sample[1], dist_thresh=threshold)
        # Update the best-fit line if we found more inliers than before
        if len(inliers) > best_count:
            best_fit = (m, b)
            best_count = len(inliers)
            points_on_line = inliers

        if len(inliers) >= inlier_thresh * len(graph):
            break

    return best_fit, points_on_line


def find_similar_triangles(src, dest):
    """
    Finds all similar triangles between two sets of points, src and dest.
    """
    src_triangles = list(itertools.combinations(src, 3))
    dest_triangles = list(itertools.combinations(dest, 3))

    similar_triangles = []
    for t1 in src_triangles:
        for t2 in dest_triangles:
            if valid_triangles(t1, t2, threshold=0.1):
                similar_triangles.append((t1, t2))

    return similar_triangles


def match_stars_ransac(G1, G2, dist_thresh=5, inlier_thresh=0.5, max_iterations=1000, validation_threshold=25,
                       triangle_sim_threshold=0.1):
    '''
    -   G1 and G2 are the input graphs, with each vertex represented as a tuple (x,y,r,b)
    -   dist_thresh is the distance threshold used to determine inliers
    -   inlier_thresh is the minimum ratio of inliers to total vertices required to terminate RANSAC
    -   max_iters is the maximum number of iterations for RANSAC
    '''

    '''
    step1 - randomly select two lines (one from each graph)
    step2 - randomly select two inliers in each graph
    step3 - find transformation between these lines
    step4 - check matching
    '''
    if len(G1) > len(G2):  # We always assume G1 is the 'smaller' image
        tmp = G1
        G1 = G2
        G2 = tmp

    # Initialize the set of best inliers found so far
    best_matchings = set()
    transformed_set = None
    best_score = 0

    for _ in range(max_iterations):
        # Compute the transformation required to match the pair of vertices
        local_score = 0
        to_sample = 3
        tpoints1 = random.sample(G1, to_sample)
        tpoints2 = random.sample(G2, to_sample)

        if not valid_triangles(tpoints1, tpoints2, triangle_sim_threshold, validation_threshold):
            continue

        T = calc_affine_transformation_matrix(src=tpoints1, dest=tpoints2)

        pt = [(x, y) for x, y, _, _ in G1]
        # Convert the point set to a NumPy array
        pt = np.array(pt)
        pt1 = np.hstack([pt, np.ones((len(pt), 1))])
        G1_transformed = np.dot(pt1, T.T)[:, :2]

        dist_dict = dict()
        matchings = []
        for i in range(len(G1_transformed)):
            for j in range(len(G2)):
                a1, b1 = G1_transformed[i]
                a1 = round(a1)
                b1 = round(b1)
                a2, b2, _, _ = G2[j]
                a2 = round(a2)
                b2 = round(b2)
                dist = dist_points(a1, b1, a2, b2)
                if dist <= validation_threshold:
                    a1, b1, _, _ = G1[i]
                    if dist_dict.get(str([a1, b1])) is None:
                        dist_dict[str([a1, b1])] = ((a1, b1), (a2, b2), dist)
                    if dist_dict[str([a1, b1])][2] > dist:
                        dist_dict[str([a1, b1])] = ((a1, b1), (a2, b2), dist)
                    # matchings.append(((a1, b1), (a2, b2)))
        for value in dist_dict.values():
            matchings.append((value[0], value[1]))
            local_score = validation_threshold - value[2]

        # Update the best set of inliers if necessary based on the score
        if local_score > best_score:
            best_matchings = matchings
            transformed_set = G1_transformed
            best_score = local_score

    # Return the set of best inliers found
    return best_matchings, transformed_set


def match_start_brute(G1, G2, dist_threshold=1000, tsim_threshold=5, validation_threshold=0.05):
    '''
    '''
    gt1 = list(itertools.combinations(G1, 3))
    gt2 = list(itertools.combinations(G2, 3))

    best_matchings = set()
    transformed_set = None
    best_score = 0
    for tri1 in gt1:
        for tri2 in gt2:
            if not valid_triangles(tri1, tri2, tsim_threshold, validation_threshold):
                continue
            local_score = 0
            T = calc_affine_transformation_matrix(src=tri1, dest=tri2)

            pt = [(x, y) for x, y, _, _ in G1]
            # Convert the point set to a NumPy array
            pt = np.array(pt)
            pt1 = np.hstack([pt, np.ones((len(pt), 1))])
            G1_transformed = np.dot(pt1, T.T)[:, :2]

            dist_dict = dict()
            matchings = []
            for i in range(len(G1_transformed)):
                for j in range(len(G2)):
                    a1, b1 = G1_transformed[i]
                    a1 = round(a1)
                    b1 = round(b1)
                    a2, b2, _, _ = G2[j]
                    a2 = round(a2)
                    b2 = round(b2)
                    dist = dist_points(a1, b1, a2, b2)
                    if dist <= dist_threshold:
                        a1, b1, _, _ = G1[i]
                        if dist_dict.get(str([a1, b1])) is None:
                            dist_dict[str([a1, b1])] = ((a1, b1), (a2, b2), dist)
                        if dist_dict[str([a1, b1])][2] > dist:
                            contained = False
                            for value in dist_dict.values():
                                if value[1][0] == a2 and value[1][1] == b2:
                                    contained = True
                                    break
                            if not contained:
                                dist_dict[str([a1, b1])] = ((a1, b1), (a2, b2), dist)


                        # matchings.append(((a1, b1), (a2, b2)))
            for value in dist_dict.values():
                matchings.append((value[0], value[1]))
                local_score += dist_threshold - value[2]

            # if len(matchings) <= 3:
            #     continue

            # Update the best set of inliers if necessary
            # if len(matchings) > len(best_matchings):
            if local_score > best_score:
                best_matchings = matchings
                transformed_set = G1_transformed
                best_score = local_score

    # Return the set of best inliers found
    return best_matchings, transformed_set


def load_graph(image_path, graph_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found, path:", )
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    W, H = gray.shape[::-1]

    parts = image_path.split('/')
    g = Graph(name=image_path, W=W, H=H)
    name = parts[-1][:-3] + 'txt'

    path = graph_path
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
                idx = i - 1
                x = int(prop[0])
                y = int(prop[1])
                b = int(prop[2])
                r = float(prop[3])
                from Vertex import Vertex
                v = Vertex(idx, x, y, r, b)
                g.add_star(v)
    return g


def convert_to_points(graph: Graph):
    points = []
    vertices = graph.get_vertice_ids()
    for id in vertices:
        vertex = graph.get(id)
        tup = (vertex.get_x(), vertex.get_y(), vertex.get_radius(), vertex.get_brightness())
        points.append(tup)
    return points


def main():

    '''
    Assumptions:

    Algorithm:
    1. Ransac for best fit with some mistake (we now have inliers and outliers)
    2. find R_2 Bases for both fits (lines V and W respectively)
    3. find Transformation matrix between V and W + a scalar for it (zoom in/out)
    4. for some points in the inlier - mutltiply by the Transformation matrix and chececk how close it is
    5. if (4) is good enough (for some mistake) and for enough points (set threshold?) then we can continue 'easily'
    '''
    print("\n\tCompiled, Loading two graphs/images to compare\n")

    graph_path = 'star_data/'
    img_path1 = 'Star_images/Formatted/IMG_3053.jpg'
    img_path2 = 'Star_images/Formatted/fr1.jpg'

    graph1 = load_graph(image_path=img_path1, graph_path=graph_path)
    if len(graph1) == 0:
        graph1 = detect_stars2(img_path1)
    graph2 = load_graph(image_path=img_path2, graph_path=graph_path)
    if len(graph2) == 0:
        graph2 = detect_stars2(img_path2)

    print("\n\tloaded images to graph objects\n")

    x1_array, y1_array = graph1.get_coordinates()
    x2_array, y2_array = graph2.get_coordinates()

    points1 = convert_to_points(graph1)
    points2 = convert_to_points(graph2)

    if len(points1) > len(points2):
        tmp = points1
        points1 = points2
        points2 = points1
        x1_array, y1_array = graph2.get_coordinates()
        x2_array, y2_array = graph1.get_coordinates()

    inlier_thresh = 1
    max_iterations = 10000

    dist_thresh = 1000
    triangle_sim_threshold = 5
    validation_threshold = 0.2

    # matchings, G1_transformed = match_stars_ransac(G1=points1, G2=points2, inlier_thresh=inlier_thresh,
    #                                                dist_thresh=dist_thresh, max_iterations=max_iterations,
    #                                                validation_threshold=validation_threshold,
    #                                                triangle_sim_threshold=triangle_sim_threshold)

    matchings, G1_transformed = match_start_brute(points1, points2, dist_threshold=dist_thresh,
                                                  tsim_threshold=triangle_sim_threshold,
                                                  validation_threshold=validation_threshold)
    if G1_transformed is None:
        print("no matches found :( ")
        return

    x_t = [x for x, _ in G1_transformed]
    y_t = [y for _, y in G1_transformed]
    matching_1 = [x for x, _ in matchings]
    matching_2 = [y for _, y in matchings]
    matching_1x = [x for x, _ in matching_1]
    matching_1y = [y for _, y in matching_1]
    matching_2x = [x for x, _ in matching_2]
    matching_2y = [y for _, y in matching_2]

    plt.subplot(2, 2, 1)
    plt.scatter(x1_array, y1_array, c='black')
    plt.scatter(matching_1x, matching_1y, c='r')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.scatter(x2_array, y2_array, c='black')
    plt.scatter(matching_2x, matching_2y, c='r')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.scatter(x2_array, y2_array, c='black')
    plt.scatter(x_t, y_t, c='green')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.scatter(x2_array, y2_array, c='black')
    plt.grid()

    plt.show()
    for match in matchings:
        print(match)


if __name__ == "__main__":
    main()
