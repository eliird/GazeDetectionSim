from threading import Thread
import numpy as np
import pykinect_azure as pykinect
import cv2
from PIL import Image
import torch
from torchvision import transforms


class Tracker:
    def __init__(self, MAX_PEOPLE=5) -> None:
        # tracking people
        self.person_dict = {i:[] for i in range(MAX_PEOPLE)}
        self.empty_frames = [0 for i in range(MAX_PEOPLE)]
        self.FLUSH_AFTER_N_FRAMES = 7
        
        image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([
                transforms.Resize((224,224)),transforms.ToTensor(),image_normalize,
        ])
        # modify the camera config
        pykinect.initialize_libraries(track_body=True)
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED # look into other modes
        
        # start the device
        self.device = pykinect.start_device(config=device_config)
        
        #start the tracker
        self.bodyTracker = pykinect.start_body_tracker()
        self.people = []
        # cv2.namedWindow('Color image with skeleton',cv2.WINDOW_NORMAL)
    
    def start(self)->None:
        run_thread = Thread(target=self.run)
        print("Starting the run thread")
        run_thread.start()
        print("Thread has started")
        # self.run() # run the thread of capturing and displaying the image
    
    
    def getFaceCrops(self, body_frame, image, padding=50):
        
        h, w, c = image.shape
        num_bodies = body_frame.get_num_bodies()
        for body_id in range(num_bodies):
            self.empty_frames[body_id] = 0
            body3d = body_frame.get_body(body_id, ).numpy()
            body2d = body_frame.get_body2d(body_id, pykinect.K4A_CALIBRATION_TYPE_COLOR).numpy()
            head_position = body3d[30] # 30 is the index for eye right, contains x, y,z , rotation quat, confidence
            
            ear_left = body2d[29]
            ear_right = body2d[31]
            neck = body2d[11] # 3
            nose = body2d[27]

            x0 = max(0, min(int(ear_right[0]), int(nose[0])) - padding)
            y0 = max(0, int(ear_right[1]) - int(1.3 * abs(ear_right[1] - neck[1])) )
            
            x1 = min(w, max(int(ear_left[0]), int(nose[0])) + padding)
            y1 = min(h, int(neck[1]))
            cv2.rectangle(image, (x0, y0), (x0 + (x1-x0), y0 + (y1-y0)), (0, 255,0), 2)
            faceImage = self.transformation(Image.fromarray(image[y0:y1, x0:x1]))
            self.person_dict[body_id].append((faceImage, head_position))
        max_people = len(self.person_dict.keys())
        
        for i in range(max_people - num_bodies):
            self.empty_frames[i] += 1
        
        for i in range(max_people):
            if self.empty_frames[i] >  self.FLUSH_AFTER_N_FRAMES:
                self.person_dict[i] = [] 
        
        process_faces  = []
        head_positions = []
        for i, face_image_batch in enumerate(self.person_dict.values()):            
            if len(face_image_batch) == 7:
                face_image_batch = [face[0] for face in face_image_batch]
                process_faces.append(torch.stack(face_image_batch).reshape(21, 224, 224))
                self.person_dict[i].pop(0)
                head_positions.append(self.person_dict[i][3][1])
            
        
        if len(process_faces) == 0:
            return torch.zeros(3,3), np.array([-1])
        
        process_faces = torch.stack(process_faces)
        return process_faces, np.array(head_positions)
        #print(process_faces.shape)
    
    def captureAndProcess(self):
            capture = self.device.update()
            
            # get capture of the tracker
            body_frame = self.bodyTracker.update()
            
            # get color image
            ret, color_image = capture.get_color_image()
            
            if not ret:
                print("could not capture the frame")
                return torch.zeros(6, 21, 224, 224), torch.zeros(6, 8)
            
            crops, head_pos = self.getFaceCrops(body_frame, color_image)
            print(crops.shape, head_pos.shape)
                
            color_skeleton = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
            cv2.imshow('Color image with Skeleton', color_skeleton)
            cv2.waitKey(1)
            
            return crops, head_pos
        
    def run(self)->None:
        # get capture
        while True:
            self.captureAndProcess()
            # Press q key to stop
            if cv2.waitKey(1) == ord('q'):  
                break
        
    def stop(self) ->None:
        pass
        