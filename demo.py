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

#import os

#dirname = os.path.dirname(__file__)

#rows_world = 1088
#cols_world = 1080

# Loop this with deifferent images in a map
imgs = ["civit_hero_banner_0_0.jpg",
        "a6de7dae-civit.png"]

#img = cv2.imread("/home/kapyla/Documents/PupilDemo/imgs/civit_hero_banner_0_0.jpg")
#img = cv2.imread("/home/kapyla/Documents/PupilDemo/imgs/a6de7dae-civit.png")
center_width = resolution[0] - 2*aruco_width - 2*pad
center_height = resolution[1] - 2*aruco_width - 2*pad
scale = np.array( [center_width/cols_pupil, center_height/rows_pupil] )
top_left = np.array([ aruco_width+pad, aruco_width+pad ])

Z = (pad, pad)
X = (resolution[0]-pad, resolution[1]-pad)
#aruco_img = cv2.circle(aruco_img, Z, radius=5, color=(0, 0, 255), thickness=-20)
#aruco_img = cv2.circle(aruco_img, X, radius=5, color=(0, 0, 255), thickness=-20)

def main():
    # Start auto-discovery of Pupil Invisible Companion devices
    network = init_network();
    network.start()

    try:
        img_idx = 0
        img = cv2.imread("./imgs/"+imgs[img_idx])
        aruco_img = create_aruco_frame()
        aruco_img = add_content(aruco_img, img)

        world_img = np.zeros((rows_pupil, cols_pupil, 3))
        gaze = (0, 0)
        gazes = []
        gazes2 = []
        
        #scale_correction = np.array([1.0878642, 1.27891882])
        #offset_correction = np.array([-16.86107453, -277.06072776])
        scale_correction = np.array([1.0, 1.0])
        offset_correction = np.array([0.0, 0.0])

        z = (0,0)
        x = (0,0)

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
                        gazes2.append( (mapped_gaze[0], mapped_gaze[1]) )
                        #mapped_gaze = mapped_gaze*scale_correction + offset_correction
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
            
            elif key == ord("w"):
                frame = cv2.cvtColor(aruco_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
 
                heatmapper = Heatmapper()
                heatmap = heatmapper.heatmap_on_img(gazes, pil_img)
                heatmap = np.array(heatmap)
                
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                cv2.imwrite("./outs/"+imgs[img_idx], heatmap)
                cv2.imshow("window", heatmap)
                cv2.waitKey(0)
                             
                img_idx = img_idx - 1
                if img_idx < 0:
                    img_idx = len(imgs)-1
                img = cv2.imread("./imgs/"+imgs[img_idx])
                aruco_img = create_aruco_frame()
                aruco_img = add_content(aruco_img, img)
                gazes = []
                gazes2= []
                
            elif key == ord("e"):
                frame = cv2.cvtColor(aruco_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
 
                heatmapper = Heatmapper()
                heatmap = heatmapper.heatmap_on_img(gazes, pil_img)
                heatmap = np.array(heatmap)
                
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                cv2.imwrite("./outs/"+imgs[img_idx], heatmap)
                cv2.imshow("window", heatmap)
                cv2.waitKey(0)
                
                img_idx = img_idx + 1
                if img_idx >= len(imgs):
                    img_idx = 0
                img = cv2.imread("./imgs/"+imgs[img_idx])
                aruco_img = create_aruco_frame()
                aruco_img = add_content(aruco_img, img)    
                gazes = []
                gazes2 = []
                    
            elif key == ord("z"):
                if len(gazes) > 0:
                    z = calc_mean_pos(gazes2);
                    print(z)
                else:
                    print("No gazes")
            elif key == ord("x"):
                if len(gazes) > 0:
                    x = calc_mean_pos(gazes2);
                    print(x)
                else:
                    print("No gazes")
            elif key == ord("c"):
                scale_correction, offset_correction = calibrate(z,x)
                print(scale_correction, offset_correction)
                gazes = []
                gazes2 = []

    # Catch interruption and disconnect gracefully
    except (KeyboardInterrupt, SystemExit):
        network.stop()


def calc_mean_pos(gazes):
    xt = 0
    yt = 0
    for g in gazes[-10:]:
        xt += g[0]
        yt += g[1]
    mean = (xt/len(gazes[-10:]), yt/len(gazes[-10:]) )
    return mean

def calibrate(z,x):
    if (x[0]-z[0]) != 0 and (x[1]-z[1]) != 0:
        scale_x = (X[0]-Z[0]) / (x[0]-z[0])
        scale_y = (X[1]-Z[1]) / (x[1]-z[1])
    else:
        print("No calibration - same calibration coordinates")
        return np.array([1,1]), np.array([0,0])
    offset_x = Z[0] - scale_x*z[0]
    offset_y = Z[1] - scale_y*z[1]
    return np.array([scale_x, scale_y]), np.array([offset_x, offset_y])
    
    
value = main()  # Execute example
cv2.destroyAllWindows()
