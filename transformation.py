from object_manipulation import project2d
from objects import Position, Rotation
from math import sin, cos, pi
import numpy as np
import pygame


class Point:
    def __init__(self, pos: Position=Position(0,0,0), rot: Rotation=Rotation(0,0,0)) -> None:
        self.transform = ObjectTransform(pos, rot)
    
    def drawObject(self, screen, offset, color):
        pygame.draw.circle(screen, color, project2d(self.transform.pos.getPos(), offset), 5)
    
    def __str__(self) -> str:
        return self.transform.__str__()


class ObjectTransform:
    def __init__(self, pos=Position(0, 0, 0), rot=Rotation(0, 0, 0)) -> None:
        self.pos = pos
        self.rot = rot
      
    def rotate_x(self, angle: float):
        rotation_x = np.matrix([
            [1, 0, 0],
            [0, cos(angle), -sin(angle)],
            [0, sin(angle), cos(angle)],
        ])
        new_pos = np.array(np.dot(rotation_x, self.pos.getPos().reshape(3,1)))
        self.pos.x, self.pos.y, self.pos.z = new_pos[0][0], new_pos[1][0], new_pos[2][0]
        self.rot.x += angle
            
    def rotate_y(self, angle: float):
        rotation_y = np.matrix([
            [cos(angle), 0, sin(angle)],
            [0, 1, 0],
            [-sin(angle), 0, cos(angle)],
        ])
        new_pos = np.array(np.dot(rotation_y, self.pos.getPos().reshape(3,1)))
        self.pos.x, self.pos.y, self.pos.z = new_pos[0][0], new_pos[1][0], new_pos[2][0]
        self.rot.y += angle
        
    def rotate_z(self, angle: float):
        rotation_z = np.matrix([
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1],
        ])
        new_pos = np.array(np.dot(rotation_z, self.pos.getPos().reshape(3,1)))
        self.pos.x, self.pos.y, self.pos.z = new_pos[0][0], new_pos[1][0], new_pos[2][0]
        self.rot.z += angle
        
    def rotate(self, rot: Rotation):
        self.rotate_x(rot.x)
        self.rotate_y(rot.y)
        self.rotate_z(rot.z)
        
    
    def inverse_rotate(self, rot: Rotation):
        self.rotate_x(-rot.x)
        self.rotate_y(-rot.y)
        self.rotate_z(-rot.z)
            
    def translate(self, vector: Position):
        self.pos.x += vector.x
        self.pos.y += vector.y
        self.pos.y += vector.z
    
    def inverse_translate(self, vector:Position):
        self.pos.x += vector.x
        self.pos.y += vector.y
        self.pos.z += vector.z
    
    
    def __str__(self) -> str:
        return f'pos: {self.pos.getPos()} \nrot: {self.rot.getRot()}'




kinect_pos = Position(1, 1, 1)
kinect_rot = Rotation(0, 0, 0)

kinect = Point(kinect_pos, kinect_rot)
print("Kinect Position Base Coordinates:")
print(kinect)

# person in kinect coordinates
person_position_kinect = Position(-1, -1, -1)
person_rotation_kinect = Rotation(0, 0, 0)
person = Point(person_position_kinect, person_rotation_kinect)
print("KINECT coordinaes:\n", person)

# gaze vector in person coordinates
gaze_vector_end_pos = Position(0, 0, 1)
gaze_vector_end_rot = gaze_vector_start_rot = Rotation(0, pi, 0) # rotation of the frame with reference to the person coordinates
gaze_vector_end = Point(gaze_vector_end_pos, gaze_vector_end_rot)

gaze_vector_start_pos = Position(0, 0, 0)
# gaze_vector_start_rot = Rotation(0, pi, 0) # rotation of the frame with reference to the person coordinates
gaze_vector_start = Point(gaze_vector_start_pos, gaze_vector_start_rot)

# gaze vector in kinect coordinates
gaze_vector_start.transform.inverse_rotate(person.transform.rot.getRotObj())
gaze_vector_start.transform.inverse_translate(person.transform.pos.getPosObj())

gaze_vector_end.transform.inverse_rotate(person.transform.rot.getRotObj())
gaze_vector_end.transform.inverse_translate(person.transform.pos.getPosObj())

# gaze vector in base coordinates
gaze_vector_start.transform.inverse_rotate(kinect.transform.rot.getRotObj())
gaze_vector_start.transform.inverse_translate(kinect.transform.pos.getPosObj())

gaze_vector_end.transform.inverse_rotate(kinect.transform.rot.getRotObj())
gaze_vector_end.transform.inverse_translate(kinect.transform.pos.getPosObj())


print("Gaze Vector End: \n",gaze_vector_end)
print("Gaze Vector Start: \n",gaze_vector_start)
