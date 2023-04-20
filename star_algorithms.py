import cv2
from Vertex import Vertex
from collections import deque



def best_star_coordinates_avarage_color(img, x, y, w, h, W, H):
    right_bound = min(W - 1, x + w)
    lower_bound = min(H - 1, y + h)
    color_sum = 0
    brightest_point = [0, 0]
    brightest_color = -1
    for i in range(y, lower_bound):
        for j in range(x, right_bound):
            color = img[i][j]
            color_sum += color
            if color > brightest_color:
                brightest_color = color
                brightest_point[0] = i
                brightest_point[1] = j

                """
                If we find 2 brightest points, we will take the one which is closer to the center
                """

            elif color == brightest_color:
                s1 = Vertex(name='1', x=brightest_point[0], y=brightest_point[1], radius=0, brightness=0)
                s2 = Vertex(name='2', x=i, y=j, radius=0, brightness=0)
                center = Vertex(name='center', x=x + int(w / 2), y=y + int(h / 2), radius=0, brightness=0)

                """
                If the new brightest point is closer to the center, take it instead of the current point
                """

                if s1.distance(center) > s2.distance(center):
                    brightest_color = color
                    brightest_point[0] = i
                    brightest_point[1] = j
    color_avg = color_sum // (w * h)
    return {"color_avg": color_avg, "center": [x + int(w / 2), y + int(h / 2)]}

def find_coordinater_radius(img, x, y, w, h, W, H):
    left_bound = 0
    right_bound = W - 1
    upper_bound = 0
    lower_bound = H - 1
    c_x = x + int(w / 2)
    c_y = y + int(h / 2)
    if img[c_y][c_x] == 0:
        for i in range(x, min(x + w, right_bound)):
            for j in range(y, min(y + h, lower_bound)):
                if img[j][i] > 0:
                    c_x = i
                    c_y = j
                    break
    up = down = left = right = 0
    changes = 4
    while changes > 0:
        changes = 0
        if c_x - left >= left and img[c_y][c_x - left] > 0:
            left += 1
            changes += 1
        if c_x + right <= right_bound and img[c_y][c_x + right] > 0:
            right += 1
            changes += 1
        if c_y - up >= upper_bound and img[c_y - up][c_x] > 0:
            up += 1
            changes += 1
        if c_y + down <= lower_bound and img[c_y + down][c_x] > 0:
            down += 1
            changes += 1

    left = c_x - left
    right = c_x + right
    dist1 = int((right - left)/2)
    up = c_y - up
    down = c_y + down
    dist2 = int((down - up)/2)
    coor_x = int((left + right)/2)
    coor_y = int((up + down)/2)
    radius = max(dist1, dist2)
    return {"radius": radius, "center": [coor_x, coor_y]}




# def find_coordinater_radius(img, x, y, w, h, W, H):
#     left_bound = 0
#     right_bound = W - 1
#     upper_bound = 0
#     lower_bound = H - 1
#     c_x = x + int(w / 2)
#     c_y = y + int(h / 2)
#     current_x, current_y = c_x, c_y
#     furthest_distance = 0
#     checked = [[False] * W for _ in range(H)]
#     stack = deque([(c_x, c_y)])
#
#     while stack:
#         x1, y1 = stack.popleft()
#         if checked[y1][x1]:
#             continue
#
#         checked[y1][x1] = True
#         color = img[y1][x1]
#         if color > 0:
#             distance = (x1 - c_x) ** 2 + (y1 - c_y) ** 2
#             if distance > furthest_distance:
#                 furthest_distance = distance
#                 current_x, current_y = x1, y1
#
#             if x1 - 1 >= left_bound and not checked[y1][x1 - 1]:
#                 stack.append((x1 - 1, y1))
#             if x1 + 1 <= right_bound and not checked[y1][x1 + 1]:
#                 stack.append((x1 + 1, y1))
#             if y1 - 1 >= upper_bound and not checked[y1 - 1][x1]:
#                 stack.append((x1, y1 - 1))
#             if y1 + 1 <= lower_bound and not checked[y1 + 1][x1]:
#                 stack.append((x1, y1 + 1))
#
#     radius = int(furthest_distance ** 0.5)
#     center = [int(current_x), int(current_y)]
#     return {"radius": radius, "center": center}
#

def find_brightness(img, x, y, radius, W, H):
    left_bound = 0
    right_bound = W - 1
    upper_bound = 0
    lower_bound = H - 1
    colors = [0]
    if y - radius - 1 >= upper_bound:
        colors.append(img[y - radius - 1][x])
    if y + radius + 1 <= lower_bound:
        colors.append(img[y + radius + 1][x])
    if x - radius - 1 >= left_bound:
        colors.append(img[y][x - radius - 1])
    if x + radius + 1 >= left_bound:
        colors.append(img[y][x - radius - 1])
    return max(colors)/255


def calculate_radius(img, x, y, w, h, W, H):
    right_bound = min(W - 1, x + w)
    lower_bound = min(H - 1, y + h)
    out_dict = best_star_coordinates_avarage_color(img, x, y, w, h, W, H)
    center = out_dict["center"]
    color_avg = out_dict["color_avg"]
    step = 1
    steps_length = []
    c_x, c_y = center
    while c_y - step >= y:
        if img[c_y - step][c_x] > color_avg:
            step += 1
        else:
            break

    steps_length.append(step)
    step = 1
    while c_y + step >= lower_bound:
        if img[c_y + step][c_x] > color_avg:
            step += 1
        else:
            break
    steps_length.append(step)
    step = 1

    while c_x - step >= x:
        if img[c_y][c_x - step] > color_avg:
            step += 1
        else:
            break
    steps_length.append(step)
    step = 1

    while c_x + step <= right_bound:
        if img[c_y][c_x + step] > color_avg:
            step += 1
        else:
            break
    steps_length.append(step)
    step = 1

    return max(steps_length) - 1


def calculate_brightness(img, x, y, w, h, W, H):
    right_bound = min(W, x + w)
    lower_bound = min(H, y + h)
    out_dict = best_star_coordinates_avarage_color(img, x, y, w, h, W, H)
    center = out_dict["center"]
    c_x, c_y = center
    radius = calculate_radius(img, x, y, w, h, W, H)
    colors = []
    colors.append(img[c_y - radius + 1][c_x])
    colors.append(img[c_y + radius - 1][c_x])
    colors.append(img[c_y][c_x - radius + 1])
    colors.append(img[c_y][c_x + radius - 1])
    return max(colors) / 255
