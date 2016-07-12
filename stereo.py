import numpy as np
import cv2


img_file1 = cv2.imread('F:\CVIP Projects\Project2\view1.jpg', 0)
#img_file2 = cv2.imread('F:\CVIP Projects\Project2\view5.jpg', 0)

c,r = img_file1.shape

def match():
    #define occlusion here
    C = np.zeros((c+1, r+1))
    M = np.zeros((r,c))    

    occlusion = 20
    
    for i in range(1, r):
        C[i][0] = i*occlusion
    
    for i in range(1, c):
        C[0][i] = i*occlusion
    
    for i in range(1, r):
        for j in range(1, c):
            min1 = C[i-1][j-1] 
            #+ C[1][i]         
            min2 = C[i-1][j] + occlusion
            min3 = C[i][j-1] + occlusion
            Cmin = np.min(min1, min2, min3)
            if (Cmin == min1):
                M[i][j] = 1
            if (Cmin == min2):
                M[i][j] = 2
            if (Cmin == min3):
                M[i][j] = 3
    
    
    p = r
    q = c
                    
    while(p!= 0 and q != 0):
        if (M[p][q] == 1):
            p -= 1 
            q -= 1
        elif (M[p][q] == 2):
            p -= 1
        elif (M[p][q] == 3):
            q -= 1

match()

