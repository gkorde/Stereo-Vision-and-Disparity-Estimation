import numpy as np
import cv2

image_view1 = cv2.imread('F:/CVIP Projects/Project2/view1.png');
image_view5 = cv2.imread('F:/CVIP Projects/Project2/view5.png');

image_gt1 = cv2.imread('F:/CVIP Projects/Project2/disp1.png',0);
image_gt5 = cv2.imread('F:/CVIP Projects/Project2/disp5.png',0);

#cv2.imshow("gt1",image_gt1)
#cv2.imshow("gt5",image_gt5)

height,width,channel = image_view1.shape

view3 = np.zeros((height,width,channel),dtype = np.uint8)

for i in range(0, height):
    for j in range(0, width):
        a=image_gt1[i][j]
        view3[i][j-a/2] = image_view1[i][j]


for i in range(0, height):
    for k in range(0, width):
        b=image_gt5[i][k]
        if (k+b/2 >= width):
            continue
        if (view3[i][k+b/2].all() == 0):
            view3[i][k+b/2] = image_view5[i][k]

cv2.imshow("View3_new", view3)             