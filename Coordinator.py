"""
1. Find the brightest point in the sub image - this is the center of the star
It should be close to (X + w, Y + h)
2. go one step in each direction and check if the color is the same as the center point - delta
    2.1 if it's not, stop
3. after we finished

"""
import copy

import cv2
import numpy as np


def fix(img):
    scale_percent = 10

    # calculate the 50 percent of original dimensions
    width = int(img.shape[0] * scale_percent / 80)
    height = int(img.shape[1] * scale_percent / 80)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(img, dsize)

    return output


def get_star_radius(img, X, Y, w, h, W, H):
    upper_lim = max(Y, 0)
    right_lim = min(X + w, W)
    lower_lim = min(Y + h, H)
    left_lim = max(X, 0)
    steps = []
    # find upper limit
    step = 1
    mid_x = X + int(w / 2)
    mid_y = Y + int(h / 2)
    mid_color = img[mid_x, mid_y]
    while mid_y - step > upper_lim:
        color = img [mid_x][mid_y - step]
        if img[mid_x - step][mid_y] < mid_color:
            steps.append(step)
            step = 1
            break
        step += 1
    if len(steps)<1:
        steps.append(step)
        step = 1
    while mid_y + step < lower_lim:
        if img[mid_x][mid_y + step] < mid_color:
            steps.append(step)
            step = 1
            break
        step += 1
    if len(steps)<2:
        steps.append(step)
        step = 1
    while mid_x - step > left_lim:
        if img[mid_x - step][mid_y] < mid_color:
            steps.append(step)
            step = 1
            break
        step += 1
    if len(steps)<3:
        steps.append(step)
        step = 1
    while mid_x + step < right_lim:
        if img[mid_x + step][mid_y] < mid_color:
            steps.append(step)
            step = 1
            break
        step += 1
    if len(steps)<4:
        steps.append(step)
        step = 1
    return max(steps)



if __name__ == "__main__":
    img = cv2.imread('Star_Images/Formatted/IMG_3057.jpg')
    img = fix(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reg_gray = copy.deepcopy(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            print('Radius of circle:', r)
    cv2.imshow("Detected Circles", gray)
    cv2.imshow("Circles", reg_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
