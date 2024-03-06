
import math
import pygame
import torch
from object_manipulation import *
from axis import Axis
from objects import Object, Position, Rotation
from tracker import Tracker
from model import GazeLSTM
import torch_directml

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

colors = [RED, GREEN, BLUE, BLACK]

class Simulator:
    def __init__(self, WIDTH=1920, HEIGHT=1080, SCALE=100, kinect_config_file='environment.txt') -> None:
        # load pygame screen
        pygame.display.set_caption("Gaze Detection")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))  
        self.scale = SCALE
        self.origin2d = [WIDTH/2, HEIGHT/2] # base reference frame in the center of the screen
        self.default_rotation = 0
        
        # load the kinect position and rotation
        handle = open(kinect_config_file, 'r')
        lines = handle.readlines()
        kinect_pos = lines[0].split('#')[0].split(',')
        kinect_pos = Position(float(kinect_pos[0]), float(kinect_pos[1]), float(kinect_pos[2]))
        kinect_rot = lines[1].split('#')[0].split(',')
        kinect_rot = Rotation(float(kinect_rot[0]), float(kinect_rot[1]), float(kinect_rot[2]))
        handle.close()
        
        self.kinect = Object(self.screen, self.origin2d, kinect_pos, kinect_rot, color=RED)
        self.kinectOrigin2d = project2d(self.kinect.getPosition(), self.origin2d, self.scale)
        # draw the base reference axis
        self.screen.fill(WHITE)
        self.drawAxis()
        
        
        # set up the tracker
        self.tracker = Tracker()
        
        
        # ML model
        self.train_device = torch_directml.device()
        self.model = GazeLSTM()
        self.model.load_state_dict(torch.load('./next_model/model_best_Gaze360.pth.tar'), map_location=torch.device('cpu'))
        self. model.eval()
        self.model = self.model.to(self.train_device)
        
        
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
        self.screen.fill(WHITE)
        self.kinect.drawObject()
        faces, head_positions = self.tracker.captureAndProcess()
        # need a more robust way to handle the not detected condition
        if faces.shape[0] == 6 or faces.shape[1] != 21:
            return
        #print(faces.shape, head_positions.shape)
        for i, person in enumerate(head_positions):
            print("Perosn: ", person[0:3])
            print("Kinect Origin:", self.kinectOrigin2d)
            person_object = Object(self.screen, self.kinectOrigin2d, Position(person[0]/100, person[1]/100, person[2]/100), Rotation(0, 0, 0), color=RED) # change the person rotation from quad to x, y, z
            person_object.drawObject(colors[i])

    def start(self):
        while True:
            # print("Checking for events")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            #print("Updating the screen")
            self.update()
            pygame.display.update()