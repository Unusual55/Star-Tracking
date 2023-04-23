import math


class Vertex:
    def __init__(self, name, x, y, radius, brightness):
        self.__name = name
        self.__x = x
        self.__y = y
        self.__radius = radius
        self.__brightness = brightness

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def fix_x(self, x):
        self.__x = (self.__x + x) // 2

    def fix_y(self, y):
        self.__y = (self.__y + y) // 2

    def get_name(self):
        return self.__name

    def set_name(self, name):
        self.__name = name

    def get_radius(self):
        return self.__radius

    def get_brightness(self):
        return self.__brightness

    def distance(self, other):
        delta_x = (other.get_x() - self.__x) ** 2
        delta_y = (other.get_y() - self.__y) ** 2
        return math.sqrt(delta_x + delta_y)

    def __str__(self):
        return str({"name": self.__name, "x": self.__x, "y": self.__y, "radius": self.__radius,
                    "brightness": self.__brightness})

    def __eq__(self, other):
        if self.__x == other.__x and self.__y == other.__y:
            return True
        return False

    def get_text_format(self):
        return f'{self.__x}, {self.__y}, {self.__radius}, {self.__brightness}'

    def __hash__(self):
        return hash(str(self))
