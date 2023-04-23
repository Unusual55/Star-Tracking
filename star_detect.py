import cv2
import numpy as np
import matplotlib.pyplot as plt
from ImageDenoiser import denoise
from star_algorithms import find_coordinater_radius, find_brightness
from Graph import Graph, Vertex
import writer as writer
import os

"""
This function will handle dark images cases, in which the detect function will not find any star.
It will make the image brighter and try to find a stars.
"""


def second_level_detection(img, image_name):
    # Loading template for star

    alpha = 5
    beta = 50

    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised_image = denoise(image_name, new_image)
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('star_template.jpg', 0)
    # changing into gray scale
    w, h = template.shape[::-1]
    W, H = gray_image.shape[::-1]

    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    threh = 0.45

    loc = np.where(result >= threh)
    g = Graph(image_name, W, H)

    for pt in zip(*loc[::-1]):
        ui_pt = (int(pt[0] + w / 2), int(pt[1] + h / 2))
        out_dict = find_coordinater_radius(gray_image, int(pt[0] + w / 2), int(pt[1] + h / 2), w, h, W, H)
        c_x, c_y = out_dict["center"]
        radius = out_dict["radius"]
        if radius == 0:
            continue
        brightness = find_brightness(gray, c_x, c_y, radius, W, H)
        if brightness == 0 :
            continue
        output = g.add_star(Vertex(name=str(len(g)), x=c_x, y=c_y, radius=radius, brightness=brightness))
    return g


def denoise_gray(image_name):
    img = cv2.imread(image_name)
    out = denoise(image_name, img)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return gray


"""
This function will get a name of an image and will return a graph based on the found stars
"""


def detect_stars2(image_name):
    img = cv2.imread(image_name)
    gray_image = denoise_gray(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Loading template for star
    template = cv2.imread('star_template.jpg', 0)
    # changing into gray scale
    w, h = template.shape[::-1]
    W, H = gray_image.shape[::-1]

    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    threh = 0.1

    loc = np.where(result >= threh)
    g = Graph(image_name, W, H)

    for pt in zip(*loc[::-1]):

        ui_pt = (int(pt[0] + w / 2), int(pt[1] + h / 2))
        out_dict = find_coordinater_radius(gray_image, int(pt[0] + w / 2), int(pt[1] + h / 2), w, h, W, H)
        c_x, c_y = out_dict["center"]
        radius = out_dict["radius"]
        if radius == 0:
            continue
        brightness = find_brightness(gray, c_x, c_y, radius, W, H)
        if brightness == 0 or brightness == 1:
            continue
        output = g.add_star(Vertex(name=str(len(g)), x=c_x, y=c_y, radius=radius, brightness=brightness))

    if len(g) == 0:
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = second_level_detection(img, image_name)

    writer.write_graph(g)

    _, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY, dst=None)
    _, threshold2 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY, dst=None)

    return g


"""
This function will get a graph as an input and will display its stars in an image
"""


def display_detected_stars(g: Graph):
    img = cv2.imread(g.get_name())
    for star in g:
        cv2.circle(img, (star.get_x(), star.get_y()), star.get_radius(), (0, 255, 255), 2)
        text = star.get_name()
        coordinates = (star.get_x(), star.get_y() + 2 * star.get_radius())
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 255)
        thickness = 2
        cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    cv2.imshow("output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# path = 'Star_Images/Formatted/'
# entries = os.listdir(path)
# for entry in entries:
#     file_path = path + entry
#     detect_stars2(file_path)

# g1 = detect_stars2('Star_Images/Formatted/IMG_3054.jpg')
# display_detected_stars(g1)
