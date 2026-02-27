import math


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])