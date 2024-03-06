from enum import Enum
import numpy as np

import pygame

from object_manipulation import project2d

class Position():
    x: float
    y:float
    z:float
    
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getZ(self):
        return self.z
    
class Rotation():
    x: float
    y:float
    z:float
    
    def __init__(self, x, y,z) -> None:
        self.x = x
        self.y = y
        self.z = z  

    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getZ(self):
        return self.z
    
class Object:
    def __init__(self, screen, origin, pos:Position, rot:Rotation, color=(0, 255, 0)) -> None:
        self.position = pos
        self.rotation = rot
        self.screen = screen
        self.color = color
        self.origin = origin
        self.drawObject()
        
    def drawObject(self, color=(0, 255, 0)):
        pygame.draw.circle(self.screen, color, project2d(np.matrix([self.position.x, self.position.y, self.position.z]),self.origin), 5)
    
    def getPosition(self):
        return np.matrix([self.position.x, self.position.y, self.position.z])   
    
    def getRotation(self):
        return self.rotation
    
    def setColor(self, color):
        self.color = color
        self.drawObject(color)
    
    def setPosition(self, pos:Position):
        self.position = pos
        self.drawObject(self.color)
    
    def setRotation(self, rot:Rotation):
        self.rotation = rot
        self.drawObject(self.color)
        
    def translate(self, dx, dy, dz):
        self.x += dx
        self.y += dy
        self.z += dz

    def inverse_translate(self, dx, dy, dz):
        self.x -= dx
        self.y -= dy
        self.z -= dz

    def rotate(self, angle_x, angle_y, angle_z):

        # Define rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angle_x), -np.sin(angle_x)],
                       [0, np.sin(angle_x), np.cos(angle_x)]])
        
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                       [0, 1, 0],
                       [-np.sin(angle_y), 0, np.cos(angle_y)]])

        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                       [np.sin(angle_z), np.cos(angle_z), 0],
                       [0, 0, 1]])

        # Combine rotation matrices
        R = np.dot(Rz, np.dot(Ry, Rx))

        # Apply rotation
        coordinates = np.array([self.x, self.y, self.z])
        rotated_coordinates = np.dot(R, coordinates)
        self.x, self.y, self.z = rotated_coordinates.tolist()

    def inverse_rotate(self, angle_x, angle_y, angle_z):
        self.rotate(-angle_x, -angle_y, -angle_z)
        