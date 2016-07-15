import numpy as np
import cv2

#part 2

img_file1 = cv2.imread('F:/CVIP Projects/Project2/view1.png', 0)
img_file2 = cv2.imread('F:/CVIP Projects/Project2/view5.png', 0)

r,c = img_file1.shape

dmap1 = np.zeros((r,c))
dmap2 = np.zeros((r,c))

def match():
    #define occlusion here
    for x in range (0, r):
        C = np.zeros((c+1, c+1))
        M = np.zeros((c+1,c+1))    
    
        occlusion = 40
        
        for i in range(1, c+1):
            C[i][0] = i*occlusion
        
        for i in range(1, c+1):
            C[0][i] = i*occlusion
        
        for i in range(1, c+1):
            for j in range(1, c+1):
                min1=C[i-1][j-1]+np.absolute(img_file1[x][i-1]-img_file2[x][j-1])
                # + matching cost         
                min2 = C[i-1][j] + occlusion
                min3 = C[i][j-1] + occlusion
                C[i][j] =min(min1, min2, min3)
                Cmin = C[i][j]
                
                if (Cmin == min1):
                    M[i][j] = 1
                if (Cmin == min2):
                    M[i][j] = 2
                if (Cmin == min3):
                    M[i][j] = 3
        
        p = c
        q = c
                        
        while(p!= 0 and q != 0):
            if (M[p][q] == 1):
                p -= 1 
                q -= 1
                dmap1[x][p] = np.absolute(p-q)
                dmap2[x][q] = np.absolute(p-q)
            elif (M[p][q] == 2):
                p -= 1
                dmap1[x][p] = np.absolute(p-q)
            elif (M[p][q] == 3):
                q -= 1
                dmap2[x][q] = np.absolute(p-q)
    
            
match()

dmap1 = dmap1/dmap1.max()
dmap2 = dmap2/dmap2.max()
#print dmap1            
cv2.imshow('Disparity1_part2', dmap1)
cv2.imshow('Disparity2_part2', dmap2)     
