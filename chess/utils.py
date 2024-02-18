from functools import lru_cache
import math


@lru_cache(1)
def hexagon_dimensions(size):
    width = 2 * size
    height = math.sqrt(3) * size

    return width, height


@lru_cache(1)
def hexagon_points(size):
    points = [None] * 6
    for i in range(6):
        angle = i * math.pi / 3
        points[i] = (size * math.cos(angle), size * math.sin(angle))
    return points


def qr_to_xy(q, r, size):
    width, height = hexagon_dimensions(size)
    x = 3 / 4 * width * q
    y = height / 2 * q + height * r
    return x, y


def xy_to_qr(x, y, size):
    width, height = hexagon_dimensions(size)
    q = 4 / 3 * x / width
    r = y / height - q / 2
    return round(q), round(r)
