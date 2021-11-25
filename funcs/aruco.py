#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import imutils

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

aruco_width = 150
pad = 20
resolution = (1920,1080)

#aruco_img = create_aruco_frame()


def find_markers(frame):
    
    projective_matrix = np.zeros((3,3))

    # detect ArUco markers in the input frame    
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    
    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
        
            # If all necessary corners found
            if all(elem in ids  for elem in [1,2,3,4]):          
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):  
                  
                    if markerID == 1:
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        tl = (int(bottomRight[0]), int(bottomRight[1]))
                    elif markerID == 3:
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        tr = (int(bottomLeft[0]), int(bottomLeft[1]))
                    elif markerID == 2:
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        bl = (int(topRight[0]), int(topRight[1]))
                    elif markerID == 4:
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        br = (int(topLeft[0]), int(topLeft[1]))
                
                rows, cols = frame.shape[:2] # HOX
                src_points = np.float32([[0,0], [0,rows-1], [cols-1,0], [cols-1,rows-1]])
                dst_points = np.float32([tl, tr, bl, br])
                projective_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
                #projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

                """
                warped = cv2.warpPerspective(frame, projective_matrix, (cols,rows))
                frame = frame+warped
                """
                
    return projective_matrix


# Create frame with 4 aruco markers
def create_aruco_frame():
    aruco_img = np.ones((resolution[1], resolution[0], 3))
    aruco_img = np.uint8(aruco_img) * 255
    aruco_img = cv2.rectangle(aruco_img, (0,0), (resolution[0],resolution[1]), (255,0,0), 2)

    img1 = cv2.aruco.drawMarker(arucoDict,1, aruco_width)
    img2 = cv2.aruco.drawMarker(arucoDict,2, aruco_width)
    img3 = cv2.aruco.drawMarker(arucoDict,3, aruco_width)
    img4 = cv2.aruco.drawMarker(arucoDict,4, aruco_width)
    
    aruco_img[pad:aruco_width+pad, pad:aruco_width+pad] = cv2.merge([img1,img1,img1])
    aruco_img[pad:aruco_width+pad, resolution[0]-aruco_width-pad:resolution[0]-pad] = cv2.merge([img2,img2,img2])
    aruco_img[resolution[1]-aruco_width-pad:resolution[1]-pad, pad:aruco_width+pad] = cv2.merge([img3,img3,img3])
    aruco_img[resolution[1]-aruco_width-pad:resolution[1]-pad, resolution[0]-aruco_width-pad:resolution[0]-pad] = cv2.merge([img4,img4,img4])
    
    return aruco_img


# Adds image to the center of the aruco image   
def add_content(aruco_img, img):
    # Resize image to max size while keeping aspect ratio
    max_width = resolution[0] - 2*aruco_width - 4*pad
    max_height = resolution[1] - 2*aruco_width - 4*pad
    img2 = imutils.resize(img, width=max_width)
    if img2.shape[0] > max_height:
        img2 = imutils.resize(img, height=max_height)
        
    height, width = img2.shape[:2]    
    corner_y = round(resolution[1]/2 - height/2)
    corner_x = round(resolution[0]/2 - width/2)
    
    aruco_img[corner_y:corner_y+height, corner_x:corner_x+width] = img2
    return aruco_img

    
#%%
aruco_img = create_aruco_frame()

"""
#%%
img = cv2.imread("/home/kapyla/Documents/PupilDemo/imgs/civit_hero_banner_0_0.jpg")
#img = cv2.imread("/home/kapyla/Documents/PupilDemo/imgs/joker.jpg")

aruco_img = create_aruco_frame()

img2 = add_content(img, aruco_img)

cv2.imshow("window", img2)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow("window", aruco_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
"""