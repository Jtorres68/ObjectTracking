# -*- coding: utf-8 -*-

"""
Created on Fri Aug  9 15:04:41 2019

@author: Daisuke Shimizu

Video tracking:
    
    Throughout video tracks green objects and creates a surrounding red box
    Also creates a second video with the following mask where everything green is white and everything else is black

"""

import cv2
import numpy as np
import time
from collections import deque
import argparse

#calculate the distance between the camera and the object
#modified 8/3/2020
dist = 0
focal = 430
pixels = 30
width = 4

def get_distance(circ, image):
    pixels = circ[1][0]
    dist = (width*focal)/pixels
    
    image = cv2.putText(image, str(dist), (110,50), cv2.FONT_HERSHEY_SIMPLEX,  
       1.0, (0, 0, 255), 1, cv2.LINE_AA)
    return image


'''
modified 2/6/2020
start
'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

'''
end
'''

camera_feed = cv2.VideoCapture(r'C:\Users\Daisuke Shimizu\Pictures\laser_test.mp4')
minArea = 500

#intialize first frame
firstFrame = None

while True:
    #read frames from camera
    _,frame = camera_feed.read()
    
    if frame is None:
        break
    
    #Convert the current frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Define the threshold for finding a green object with hsv
    greenLower = (54, 143, 176)  #original: 35, 21, 50
    greenUpper = (70, 255, 255) #original: 80, 255, 255
    
    '''
    modified 2/6/2020
    start
    '''
    pts = deque(maxlen=args["buffer"])
    '''
    end
    '''

    #Create a binary image, where anything green appears white and everything else is black
    #then blur the image
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.GaussianBlur(mask, (81, 81), 0)

    #Get rid of background noise using erosion and fill in the holes using dilation and erode the final image on last time
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.erode(mask,element, iterations=2)
    mask = cv2.dilate(mask,element,iterations=2)
    mask = cv2.erode(mask,element)
    
   #assign the current frame as the first frame 
    if firstFrame is None:
        firstFrame = mask
        continue

    #compute the abs difference between frames
    frameDiff = cv2.absdiff(firstFrame, mask)
    thresh = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)[1]
    
    #dilate the threshold to fill the holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    #Create Contours of the moving objects
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        
        '''
        modified 2/6/2020
        start
        '''
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        print(center)
        
        #proceed if radius meets min size
        #if radius > 5:
        #draw circle and centroid
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 3)
        circ_cent = cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        
        #update points queue
        pts.appendleft(center)
        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
            
            frame = get_distance(circ_cent, frame)
            
        
        '''
        end
        '''
   
        
    
    #return to None
    #firstFrame = None
    #time.sleep(0.5)
    
    #Show the original camera feed with a bounding box overlayed
    cv2.imshow('frame',frame)
    #Show the contours in a seperate window
    cv2.imshow('thresh',thresh)
    #Use this command to prevent freezes in the feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera_feed.release()
cv2.destroyAllWindows()


