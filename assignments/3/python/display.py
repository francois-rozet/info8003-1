#!/usr/bin/env python

import numpy as np
import pygame as pg

from functools import lru_cache as cache
from math import atan2, degrees

from domain import *


# Globals

SIZE = 400

PINE = '../resources/pine_tree.png'
CAR = '../resources/car.png'


# Functions

def x2p(x: int) -> float:
    return 2 * (x / SIZE) - 1


def p2x(p: float) -> int:
    return round(SIZE * (1 + p) / 2)


def h2y(h: float) -> int:
    return round(SIZE * (1 - h) / 2)


@cache()
def load(filename: str) -> pg.Surface:
    img = pg.image.load(filename)

    return img


@cache()
def background() -> pg.Surface:
    surf = pg.Surface((SIZE, SIZE))

    # Sky, hill & road

    for x in range(SIZE):
        z = h2y(hill(x2p(x)))

        for y in range(SIZE):
            if y < z:
                color = (64, 163, 191)  # sky
            elif y > z:
                color = (64, 191, 114)  # hill
            else:
                color = (0, 0, 0)  # road

            surf.set_at((x, y), color)

    # Trees

    pine = load(PINE)
    w, h = pine.get_rect().size

    for p in [-0.5, 0.5]:
        x, y = p2x(p), h2y(hill(p))
        surf.blit(pine, (x - w // 2, y - h))

    return surf


def draw(p: float, s: float) -> pg.Surface:
    surf = background().copy()

    # Car

    car = load(CAR)
    w, h = car.get_rect().size

    x, y = p2x(p), h2y(hill(p))
    angle = degrees(atan2(hill_prime(p), 1))

    car = pg.transform.rotate(car, angle)

    surf.blit(car, car.get_rect(center=(x, y - h // 2)))

    # Speedometer

    location = (SIZE - 70, SIZE - 70)
    width, height = 30, 50
    thickness = 3

    ## Line

    rect = pg.Rect(location, (width, thickness))
    surf.fill((0, 0, 0), rect)

    ## Speed

    norm = min(abs(s), 3) / 3
    red = int(norm * 255)
    green = 255 - red
    height = int(norm * height)

    rect = pg.Rect(location, (width, height))

    if s < 0:
        rect = rect.move(0, thickness)
    else:
        rect = rect.move(0, -height)

    surf.fill((red, green, 0), rect)

    return surf


def surf2img(surf: pg.Surface) -> np.ndarray:
    return pg.surfarray.pixels3d(surf).swapaxes(0, 1).copy()
