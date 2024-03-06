import numpy as np


class Axis:
    def __init__(self,
                 origin=[0 ,0 ,0],
                 x_axis=[1, 0, 0],
                 y_axis=[0, 1, 0],
                 z_axis=[0, 0, 1]) -> None:
        self.origin = np.matrix(origin)
        self.x_axis = np.matrix(x_axis)
        self.y_axis = np.matrix(y_axis)
        self.z_axis = np.matrix(z_axis)
        # self.rotation = 10
    
    def set_origin(self, origin: list):
        self.origin = np.matrix(origin)
    
    def get_origin(self):
        return self.origin
    
    def get_x_axis(self):
        return self.x_axis
    
    def get_y_axis(self):
        return self.y_axis
    
    def get_z_axis(self):
        return self.z_axis
     
        