from math import cos, sin, tan
import numpy as np
import pygame

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

def rotate_x(point:np.array, angle: float):
    rotation_x = np.matrix([
        [1, 0, 0],
        [0, cos(angle), -sin(angle)],
        [0, sin(angle), cos(angle)],
    ])
    return np.dot(rotation_x, point.reshape((3,1)))
    
    
def rotate_y(point:np.array, angle: float):

    rotation_y = np.matrix([
        [cos(angle), 0, sin(angle)],
        [0, 1, 0],
        [-sin(angle), 0, cos(angle)],
    ])
    return np.dot(rotation_y, point.reshape((3,1)))
    

def rotate_z(point, angle: float):
    rotation_z = np.matrix([
        [cos(angle), -sin(angle), 0],
        [sin(angle), cos(angle), 0],
        [0, 0, 1],
    ])
    return np.dot(rotation_z, point.reshape((3, 1)))


def project2d(point, origin2d ,scale=100):
    projection_matrix = np.matrix([
        [1, 0, 0],
        [0, 1, 0]
    ])
    projected2d = np.dot(projection_matrix, point.reshape((3, 1)))
    
    x = int(projected2d[0][0] * scale) + origin2d[0]
    y = int(projected2d[1][0] * scale) + origin2d[1]
    return (x, y)
    
def connect_points_2d(screen, pointA, pointB, color=BLACK):
    pygame.draw.line( screen, color, (pointA[0], pointA[1]), (pointB[0], pointB[1]))


def connect_points_3d(screen, pointA, pointB, origin2d, color=BLACK):
    pointA_2d = project2d(pointA, origin2d)
    pointB_2d = project2d(pointB, origin2d)
    connect_points_2d(screen, pointA_2d, pointB_2d, color)


def draw_circle_2d(screen, pos, size=5, color=RED):
    pygame.draw.circle(screen, color, (pos[0], pos[1]), size)
    
def draw_circle_3d(screen, pos, origin2d, size=5, color=RED):
    pygame.draw.circle(screen, color, project2d(pos, origin2d), size)
    
