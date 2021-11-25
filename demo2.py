#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import time

import sys

import cv2
import numpy as np

from heatmappy import Heatmapper
from PIL import Image

# https://github.com/pupil-labs/pyndsi/tree/v1.0
#import ndsi  # Main requirement

#import imutils
#from PIL import Image
from funcs.aruco import *
from funcs.pupil import *


#rows_world = 1088
#cols_world = 1080

# Loop this with deifferent images in a map
img = cv2.imread("/home/kapyla/Documents/PupilDemo/imgs/civit_hero_banner_0_0.jpg")
aruco_img = add_content(img)
center_width = resolution[0] - 2*aruco_width - 2*pad
center_height = resolution[1] - 2*aruco_width - 2*pad
scale = np.array( [center_width/cols_pupil, center_height/rows_pupil] )
top_left = np.array([ aruco_width+pad, aruco_width+pad ])

correction = np.array([1,1])

def main():
    # Start auto-discovery of Pupil Invisible Companion devices
    network = init_network();
    network.start()

    try:
        world_img = np.zeros((rows_pupil, cols_pupil, 3))
        gaze = (0, 0)
        gazes = []
        projective_matrix = np.zeros((3,3))
        
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", aruco_img)
        key = cv2.waitKey(1)

        # Event loop, runs until interrupted
        while network.running:
            # Check for recently connected/disconnected devices
            if network.has_events:
                network.handle_event()

            # Fetch new data and update if necessary
            world_img_temp, gaze_temp = fetch_sensor_data()
            
            # New image -> try to find markers
            if world_img_temp.any():
                world_img = np.uint8(world_img_temp)
                projective_matrix = find_markers(world_img)
                
                """
                # Show world video with gaze overlay
                frame = aruco_img
                cv2.circle(
                        world_img,
                        gaze,
                        40, (0, 0, 255), 4
                        )
                
                cv2.imshow("Pupil Invisible - Live Preview", world_img)
                """
                
                
            # New gaze position and matrix exists->        
            if gaze_temp != (0,0):
                gaze = gaze_temp
                if projective_matrix.any():
                    point_in_pupil = np.array([gaze[0], gaze[1], 1]) # Homogenous coordinate
                    #point_in_frame = np.matmul( np.linalg.inv(projective_matrix), point_in_pupil )
                    point_in_frame = np.matmul( projective_matrix, point_in_pupil )
                    point_in_frame = point_in_frame / point_in_frame[2]
                    #print(point_in_frame)
                    
                    #if 0 < point_in_frame[0] < cols_pupil-1 and 0 < point_in_frame[1] < rows_pupil-1:  
                    if 1:
                        mapped_gaze = top_left + point_in_frame[0:2] * scale
                        gazes.append( (mapped_gaze[0], mapped_gaze[1]) )
                        frame = aruco_img.copy()
                        # Show world video with gaze overlay
                        cv2.circle(
                                frame,
                                ( int(mapped_gaze[0]) , int(mapped_gaze[1]) ),
                                40, (0, 0, 255), 
                                )
                        
                        cv2.imshow("window", frame)
                        
                
            #cv2.imshow("window", aruco_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                frame = cv2.cvtColor(aruco_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
 
                heatmapper = Heatmapper()
                heatmap = heatmapper.heatmap_on_img(gazes, pil_img)
                heatmap = np.array(heatmap)
                
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

                cv2.imshow("window", heatmap)
                key = cv2.waitKey(0)
            
                network.stop()
                return projective_matrix

    # Catch interruption and disconnect gracefully
    except (KeyboardInterrupt, SystemExit):
        network.stop()




value = main()  # Execute example
cv2.destroyAllWindows()
