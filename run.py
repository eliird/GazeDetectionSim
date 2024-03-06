import pygame
import numpy as np
from math import *
from object_manipulation import *
from axis import Axis

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

WIDTH, HEIGHT = 800, 600

pygame.display.set_caption("Gaze Detection")
screen = pygame.display.set_mode((WIDTH, HEIGHT))

scale = 100

origin2d = [WIDTH/2, HEIGHT/2]  # x, y

axis = Axis()
yino_position = np.matrix([0, 0, -2])
object_position = np.matrix([-2, 1, 1])


def drawAxis(screen, axis:Axis, origin2d, angle = 45):
    point1 = axis.get_x_axis()
    point2 = axis.get_y_axis()
    point3 = axis.get_z_axis()

    rotatedPoint1 = rotate_y(point1, angle)
    # rotatedPoint1 = rotate_x(rotatedPoint1, angle)
    # rotatedPoint1 = rotate_y(rotatedPoint1, angle)
    
    rotatedPoint2 = rotate_y(point2, angle)
    # rotatedPoint2 = rotate_x(rotatedPoint2, angle)
    # rotatedPoint2 = rotate_y(rotatedPoint2, angle)
    
    rotatedPoint3 = rotate_y(point3, angle)
    # rotatedPoint3 = rotate_x(rotatedPoint3, angle)
    # rotatedPoint3 = rotate_y(rotatedPoint3, angle)
    
    connect_points_3d(screen, axis.get_origin(), rotatedPoint1, origin2d, BLUE) # x-axis
    connect_points_3d(screen, axis.get_origin(), rotatedPoint2, origin2d, GREEN) # y-axis
    connect_points_3d(screen, axis.get_origin(), rotatedPoint3, origin2d, RED) # z- axis



def update(angle):
    screen.fill(WHITE)
    # connect_points_3d(screen, np.matrix([0, 0, 0]), np.matrix((0, 1, 0)), origin2d)
    # angle = 10
    drawAxis(screen, axis, origin2d, angle)
    _yino_position = rotate_y(yino_position, angle)
    _object_position = rotate_y(object_position, angle)
    draw_circle_3d(screen, _yino_position, origin2d)
    draw_circle_3d(screen, _object_position, origin2d)
    
#initialize
clock = pygame.time.Clock()
angle = 0
while True:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()

    # update stuff
    update(angle)
    angle += 0.01    
    pygame.display.update()