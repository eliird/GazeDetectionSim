import pygame
import numpy as np
from math import *
from object_manipulation import *


WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

WIDTH, HEIGHT = 800, 600

pygame.display.set_caption("Gaze Detection")
screen = pygame.display.set_mode((WIDTH, HEIGHT))

scale = 100

circle_pos = [WIDTH/2, HEIGHT/2]  # x, y



cube = []

# all the cube vertices
cube.append(np.matrix([-1, -1, 1]))
cube.append(np.matrix([1, -1, 1]))
cube.append(np.matrix([1,  1, 1]))
cube.append(np.matrix([-1, 1, 1]))
cube.append(np.matrix([-1, -1, -1]))
cube.append(np.matrix([1, -1, -1]))
cube.append(np.matrix([1, 1, -1]))
cube.append(np.matrix([-1, 1, -1]))


projected_cube = [
    [n, n] for n in range(len(cube))
]


def connect_points(i, j, points):
    pygame.draw.line(
        screen, BLACK, (points[i][0], points[i][1]), (points[j][0], points[j][1]))


def update(angle):
    screen.fill(WHITE)
 
    i = 0
    for vertex in cube:
        rotated3d = rotate_z(angle, vertex)
        rotated3d = rotate_y(angle, rotated3d)
        rotated3d = rotate_x(angle, rotated3d)
        
        projected2d = project2d(rotated3d)

        x = int(projected2d[0][0] * scale) + circle_pos[0]
        y = int(projected2d[1][0] * scale) + circle_pos[1]

        projected_cube[i] = [x, y]
        draw_circle(screen, (x, y), 5, RED)
        i += 1

    for p in range(4):
        connect_points(p, (p+1) % 4, projected_cube)
        connect_points(p+4, ((p+1) % 4) + 4, projected_cube)
        connect_points(p, (p+4), projected_cube)
    
    
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