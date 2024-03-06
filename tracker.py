from threading import Thread
import pykinect_azure as pykinect
import cv2

class Tracker:
    def __init__(self) -> None:
        
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
        faces = []
        head_positions = []
        h, w, c = image.shape
        num_bodies = body_frame.get_num_bodies()
        for body_id in range(num_bodies):
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
            
        
        
    def run(self)->None:
        # get capture
        while True:
            capture = self.device.update()
            
            # get capture of the tracker
            body_frame = self.bodyTracker.update()
            
            # get color image
            ret, color_image = capture.get_color_image()
            
            if not ret:
                print("could not capture the frame")
                continue
            
            self.getFaceCrops(body_frame, color_image)
            color_skeleton = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
            cv2.imshow('Color image with Skeleton', color_skeleton)
            
            # Press q key to stop
            if cv2.waitKey(1) == ord('q'):  
                break
        
    def stop(self) ->None:
        pass
        