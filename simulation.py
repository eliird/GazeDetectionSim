
import math
import time
from matplotlib import pyplot as plt
# import pygame
import torch
from object_manipulation import *
from axis import Axis
from objects import Object, Position, Rotation
from tracker import Tracker
from model import GazeLSTM
from transformation import Point

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

colors = [RED, GREEN, BLUE, BLACK, BLUE]


class Simulator:
    def __init__(self, WIDTH=1600, HEIGHT=1000, SCALE=100, kinect_config_file='environment.txt') -> None:
        # load pygame screen
        # pygame.display.set_caption("Gaze Detection")
        # self.screen = pygame.display.set_mode((WIDTH, HEIGHT))  
        # self.screen.fill(WHITE)
        
        self.scale = SCALE
        self.origin = Point(Position(0, 0, 0), Rotation(0, 0, 0))
        # self.origin2d = [WIDTH/2, HEIGHT/2] # base reference frame in the center of the screen
        
        # load the kinect position and rotation
        handle = open(kinect_config_file, 'r')
        lines = handle.readlines()
        kinect_pos = lines[0].split('#')[0].split(',')
        kinect_pos = Position(float(kinect_pos[0]), float(kinect_pos[1]), float(kinect_pos[2]))
        kinect_rot = lines[1].split('#')[0].split(',')
        kinect_rot = Rotation(float(kinect_rot[0]), float(kinect_rot[1]), float(kinect_rot[2]))
        handle.close()
        
        self.kinect = Point(kinect_pos, kinect_rot) # Object(self.screen, self.origin2d, kinect_pos, kinect_rot, color=RED)
        # self.kinectOrigin2d = project2d(self.kinect.transform.pos.getPos(), self.origin2d, self.scale)
        # draw the base reference axis
        
        # self.drawAxis()
        
        # set up the tracker
        self.tracker = Tracker()
        
        # ML model
        self.train_device = 'cuda' # torch_directml.device()
        model = GazeLSTM()
        self.model = torch.nn.DataParallel(model)
        checkpoint = torch.load('./next_model/model_best_Gaze360.pth.tar', map_location=torch.device('cuda'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self. model.eval()
        # self.model = self.model.to(self.train_device)

        self.fig = plt.figure()
        self.ax  = plt.axes(projection='3d')

        
    def quaternion_to_euler(self, q):
        # Extract quaternion components
        w, x, y, z = q

        # Calculate Euler angles
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = math.asin(2 * (w * y - z * x))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return (roll, pitch, yaw)


    def spherical2cartesial(self, x):    
        output = torch.zeros(x.size(0),3)
        output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
        output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
        output[:,1] = torch.sin(x[:,1])
        return output  

        
    def drawAxis(self, position=np.matrix([0, 0, 0]), rotation=[0.0, 0.0, 0.0]):
        axis = Axis()
        x_dir = axis.get_x_axis()
        y_dir = axis.get_y_axis()
        z_dir = axis.get_z_axis()
        
        x_dir_rot = rotate_x(x_dir, rotation[0])
        x_dir_rot = rotate_y(x_dir_rot, rotation[1])
        x_dir_rot = rotate_z(x_dir_rot, rotation[2])
        
        y_dir_rot = rotate_x(y_dir, rotation[0])
        y_dir_rot = rotate_y(y_dir_rot, rotation[1])
        y_dir_rot = rotate_z(y_dir_rot, rotation[2])
        
        z_dir_rot = rotate_x(z_dir, rotation[0])
        z_dir_rot = rotate_y(z_dir_rot, rotation[1])
        z_dir_rot = rotate_z(z_dir_rot, rotation[2])
        
        connect_points_3d(self.screen, position, x_dir_rot, self.origin2d, BLUE) # x-axis
        connect_points_3d(self.screen, position, y_dir_rot, self.origin2d, GREEN) # y-axis
        connect_points_3d(self.screen, position, z_dir_rot, self.origin2d, RED) # z- axis

    def update(self):
        # draw the position of the kinect
        # self.screen.fill(WHITE)
        # self.kinect.drawObject(self.screen, self.origin2d, GREEN)
        start_time = time.time()
        faces, head_positions = self.tracker.captureAndProcess()
        
        # need a more robust way to handle the not detected condition
        if faces.shape[0] == 6 or faces.shape[1] != 21:
            return
      
        gazeVectors, _ = self.model(faces.to(self.train_device))
        gazeVectors = gazeVectors.detach().cpu()
        gazeVectors = self.spherical2cartesial(gazeVectors)
        gazeVectors = gazeVectors.numpy()
        
        for i, person in enumerate(head_positions):
            
            gaze_vector = gazeVectors[i]
            # gaze_vector[1] = -gaze_vector[1]
            gaze_vector[0] = -gaze_vector[0]
            # person position and rotaion in kinect coordinates
            person_position =  Position(person[0]/self.scale, person[1]/self.scale, person[2]/self.scale)
            euler_rotation = self.quaternion_to_euler(person[3:7])
            person_rotation = Rotation(euler_rotation[0], euler_rotation[1], euler_rotation[2])
            person = Point(person_position, person_rotation)        
            
            # eye coordinate in person coordinates
            eye_start = Point(Position(0, 0, 0), Rotation(0, 0, -math.pi))
            eye_end = Point(Position(gaze_vector[0], gaze_vector[1], gaze_vector[2]))
            
            # print("Start eye coordinates: ", eye_start.transform.pos.x , eye_start.transform.pos.y, eye_start.transform.pos.z)
            # print("End eye coordinates: ", eye_end.transform.pos.x , eye_end.transform.pos.y, eye_end.transform.pos.z)
            # print('************')
            # # gazedPoint in person coordinates
            eye_start.transform.inverse_rotate(eye_start.transform.rot.getRotObj())
            eye_end.transform.inverse_rotate(eye_start.transform.rot.getRotObj())
            
            # print("Start Person Coordinates:",eye_start.transform.pos.x, eye_start.transform.pos.y, eye_start.transform.pos.z)
            # print("End Person Coordinates:",eye_end.transform.pos.x, eye_end.transform.pos.y, eye_end.transform.pos.z)
            # print('************')
            
            # gazedPoint in kinect coordinates
            # eye_start.transform.inverse_rotate(person.transform.rot.getRotObj())
            # eye_end.transform.inverse_rotate(person.transform.rot.getRotObj())
            eye_start.transform.inverse_translate(person.transform.pos.getPosObj())
            eye_end.transform.inverse_translate(person.transform.pos.getPosObj())
            # print("Start Kinect Coordinates:",eye_start.transform.pos.x, eye_start.transform.pos.y, eye_start.transform.pos.z)
            # print("End Kinect Coordinates:",eye_end.transform.pos.x, eye_end.transform.pos.y, eye_end.transform.pos.z)
            # print('************')

            # gazedPoint in Base Coordinates
            eye_start.transform.inverse_rotate(self.kinect.transform.rot.getRotObj())
            eye_end.transform.inverse_rotate(self.kinect.transform.rot.getRotObj())
            eye_start.transform.inverse_translate(self.kinect.transform.pos.getPosObj())
            eye_end.transform.inverse_translate(self.kinect.transform.pos.getPosObj())
            # print("Start Base Coordinates:",eye_start.transform.pos.x, eye_start.transform.pos.y, eye_start.transform.pos.z)
            # print("End Base Coordinates:",eye_end.transform.pos.x, eye_end.transform.pos.y, eye_end.transform.pos.z)
            # print('************')
            
            #plotting the results
            
            self.plotPoint(self.kinect)
            # self.plotLine(eye_start, eye_end)
            self.plotPoint(eye_start)
            # self.plotLine(eye_end, eye_start)
            self.arrowed_line(eye_start, eye_end)
            self.ax.set_xlim((-20, 20))
            self.ax.set_ylim((-20, 20))
            self.ax.set_zlim((-20, 20))
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            plt.draw()
            plt.pause(0.001)
            
        self.ax.clear()
        end_time = time.time()
        print("FPS: ", 1/(end_time - start_time))
            # self.plotPoint(self.kinect, ax)
            # 2D gaze points
            # eye_start2D = project2d(eye_start.transform.pos.getPos(), self.origin2d)
            # eye_end2D = project2d(eye_end.transform.pos.getPos(), self.origin2d)            
            
            # Draw the two points and connect them with the line
            # eye_start.drawObject(self.screen, self.origin2d, BLACK)
            # eye_end.drawObject(self.screen, self.origin2d, RED) 
    
    def arrowed_line(self, pointA: Point, pointB: Point, arrow_label=None, arrow_color='red', arrow_alpha=1.0):
        self.ax.plot([pointA.transform.pos.x, pointB.transform.pos.x ],
                     [pointA.transform.pos.y, pointB.transform.pos.y],
                     [pointA.transform.pos.z, pointB.transform.pos.z])
        
        arrow_start = np.array([pointA.transform.pos.x, pointA.transform.pos.y, pointA.transform.pos.z])
        arrow_end = np.array([pointA.transform.pos.x, pointA.transform.pos.y, pointA.transform.pos.z])
        arrow_vector = arrow_end - arrow_start
        
        self.ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2], arrow_vector[0], arrow_vector[1], arrow_vector[2], length=200, normalize=False, color=arrow_color, label=arrow_label)
    
    
    def plotLine(self, pointA: Point, pointB: Point):
        self.ax.plot([5*pointA.transform.pos.x, pointB.transform.pos.x ],
                     [5*pointA.transform.pos.y, pointB.transform.pos.y],
                     [5*pointA.transform.pos.z, pointB.transform.pos.z])
           
    def plotPoint(self, point:Point):
        self.ax.scatter(point.transform.pos.x, point.transform.pos.y, point.transform.pos.z)
    
    
    def start(self):
        while True:
            self.update()