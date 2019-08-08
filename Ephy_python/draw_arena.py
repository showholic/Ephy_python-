# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:57:52 2019

@author: Qixin
"""
#Note you have to draw the initial point as the upper left point of the rectangle
#import sys
#sys.path=['', 'C:\\Users\\Qixin\\.conda\\envs\\ephy\\python37.zip', 'C:\\Users\\Qixin\\.conda\\envs\\ephy\\DLLs', 'C:\\Users\\Qixin\\.conda\\envs\\ephy\\lib', 'C:\\Users\\Qixin\\.conda\\envs\\ephy', 'C:\\Users\\Qixin\\.conda\\envs\\ephy\\lib\\site-packages']
import cv2
import glob

drawing = False # True if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
def set_arena(img):   
    # mouse callback function
    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode, overlay, output, alpha,x2,y2
        overlay = img.copy()
        output = img.copy()
        alpha = 0.5
    
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
    
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.rectangle(overlay, (ix, iy), (x, y), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, img2)
                    cv2.imshow('image', img2)
                else:
                    cv2.circle(overlay, (x,y),5,(0,0,255),-1)
    
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv2.rectangle(overlay, (ix, iy), (x, y), (0, 255, 0), -1)
                cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, img)
                x2 , y2 = x, y
                
                
                
            else:
                cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
    
    
    ##img = np.zeros((512, 512, 3), np.uint8)
    # Get our image
    img2 = img.copy()
    
    #make cv2 windows, set mouse callback
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    
    while(1):
        cv2.imshow('image', img2)    
        # This is where we get the keyboard input
        # Then check if it's "m" (if so, toggle the drawing mode)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break    
    cv2.destroyAllWindows()
    
    coordll=(ix,iy)
    r_height=abs(y2-iy)
    r_width=abs(x2-ix)

    return coordll,r_height,r_width
 
if __name__=='__main__':
    filepath=r'C:\Users\Qixin\XuChunLab\nexdata\192043'
    samplevideo=sorted(glob.glob(filepath+'\*avi'))[0]
    vidcap=cv2.VideoCapture(samplevideo)
    success,img = vidcap.read()  
    coordll,r_height,r_width=set_arena(img)
    x=(coordll[0], coordll[0]+r_width)
    y=(coordll[1], coordll[1]+r_height)