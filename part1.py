import numpy as np
import cv2

#reading both the images
image_view1 = cv2.imread('F:/CVIP Projects/Project2/view1.png',0);
image_view5 = cv2.imread('F:/CVIP Projects/Project2/view5.png',0);

def threeblock():
    
    view1_padded = cv2.copyMakeBorder(image_view1,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    view5_padded = cv2.copyMakeBorder(image_view5,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    
    image_gt1 = cv2.imread('F:/CVIP Projects/Project2/disp1.png',0);
    image_gt5 = cv2.imread('F:/CVIP Projects/Project2/disp5.png',0);

    
    height, width = view1_padded.shape
    
    dmap1 = np.zeros((height,width))
    dmap2 = np.zeros((height,width))
    
    const_view1 = np.zeros((height-1,width-1))
    const_view5 = np.zeros((height-1,width-1))

    
    # calculating Disparity Maps for View1
    for i in range(1,height-1):
        for j in range(1,width-1):
            view1_slice = view1_padded[i-1:i+2,j-1:j+2]
            min_list = []
            for k in range(j-75,j):
                if (k <1 ):
                    k=1
                view5_slice = view5_padded[i-1:i+2,k-1:k+2]
                view1_diff = np.subtract(view1_slice,view5_slice)
                view1_square = np.multiply(view1_diff,view1_diff)
                view1_sum = np.sum(view1_square)
                min_list.append([view1_sum,k])
            
            min_list.sort()
            temp = min_list[0]
            dmap1[i][j] = j-temp[1]
    
    dmap1_img = dmap1/dmap1.max()
    cv2.imshow("DMAP1_3x3",dmap1_img)
    
    
    # calculating Disparity Maps for View5
    for i in range(1,height-1):
        for j in range(1,width-1):
            view5_slice = view5_padded[i-1:i+2,j-1:j+2]
            min_list = []
            for k in range(j+100,j,-1):
                if (k >= width-1 ):
                    k=width-2
                view1_slice = view1_padded[i-1:i+2,k-1:k+2]
                #print i,j,k
                #print view1_slice
                view5_diff = np.subtract(view1_slice,view5_slice)
                view5_square = np.multiply(view5_diff,view5_diff)
                view5_sum = np.sum(view5_square)
                min_list.append([view5_sum,k])
                
            min_list.sort()
            temp = min_list[0]
            dmap2[i][j] = temp[1]-j
    
    dmap2_img = dmap2/dmap2.max()
    cv2.imshow("DMAP2_3x3",dmap2_img)
    
    dmap1 = dmap1[1:height-1,1:width-1]
    dmap2 = dmap2[1:height-1,1:width-1]

    
    ### MSE calculation loop ###
    # view1`````
    mse_diff = np.subtract(image_gt1,dmap1)
    mse_square = np.multiply(mse_diff,mse_diff)
    mse_square_sum = np.sum(mse_square)
    mse_view1 = mse_square_sum/(height*width)
    print "3x3:MSE value for View1 is:",mse_view1

    # view5
    mse_diff = np.subtract(image_gt5,dmap2)
    mse_square = np.multiply(mse_diff,mse_diff)
    mse_square_sum = np.sum(mse_square)
    mse_view5 = mse_square_sum/(height*width)
    print "3x3:MSE value for View5 is:",mse_view5

            
    #setting 0's in ground truth
    list_zeros = np.where(dmap1==0)
    list_zeros_length = len(list_zeros)
    for i in range(0,list_zeros_length):
        x=list_zeros[0][i]
        y=list_zeros[1][i]
        image_gt1[x][y]=0
    
    list_zeros = np.where(dmap2==0)
    list_zeros_length = len(list_zeros)
    for i in range(0,list_zeros_length):
        x=list_zeros[0][i]
        y=list_zeros[1][i]
        image_gt5[x][y]=0
            
    for i in range(0,height-2):
        for j in range(0,width-2):
            disparity = dmap1[i][j]
            temp = np.absolute(j-disparity)
            if (dmap2[i][temp] != disparity):
                dmap2[i][temp]=0
                dmap1[i][j]=0
    
    for i in range(0,height-2):
        for j in range(0,width-2):
            disparity = dmap2[i][j]
            temp = np.absolute(j+disparity)
            if (dmap1[i][temp] != disparity):
                dmap1[i][temp]=0
                dmap2[i][j]=0
    
    temp_dmap1 = dmap1
    temp_dmap2 = dmap2 
    
    const_view1 = temp_dmap1/temp_dmap1.max()
    const_view5 = temp_dmap2/temp_dmap2.max()
    
    cv2.imshow("Consitency_Map1_3x3",const_view1)
    cv2.imshow("Consitency_Map2_3x3",const_view5)
    
    
    for i in range(0, height-2):
        for j in range(0, width-2):
            if (dmap1[i][j] == 0):
                image_gt1[i][j] = 0
            if (dmap2[i][j] == 0):
                image_gt5[i][j] = 0    
                    
     
    
    ### MSE calculation loop ###
    # view1
    mse_diff = np.subtract(image_gt1,dmap1)
    mse_square = np.multiply(mse_diff,mse_diff)
    mse_square_sum = np.sum(mse_square)
    mse_view1 = mse_square_sum/(height*width)
    print "3x3:MSE value for View1 after consistency check is:",mse_view1

    # view5
    mse_diff = np.subtract(image_gt5,dmap2)
    mse_square = np.multiply(mse_diff,mse_diff)
    mse_square_sum = np.sum(mse_square)
    mse_view5 = mse_square_sum/(height*width)
    print "3x3:MSE value for View5 after Consistency check is:",mse_view5

    
    

def nineblock():
    view1_padded = cv2.copyMakeBorder(image_view1,4,4,4,4,cv2.BORDER_CONSTANT,value=0)
    view5_padded = cv2.copyMakeBorder(image_view5,4,4,4,4,cv2.BORDER_CONSTANT,value=0)
    
    image_gt1 = cv2.imread('F:/CVIP Projects/Project2/disp1.png',0);
    image_gt5 = cv2.imread('F:/CVIP Projects/Project2/disp5.png',0);
    
    height, width = view1_padded.shape
    height_orig,width_orig = image_view1.shape
    print "Original",height_orig,width_orig
    print "Padded",height,width
    
    dmap1 = np.zeros((height,width))
    dmap2 = np.zeros((height,width))
    
    
    const_view1 = np.zeros((height-4,width-4))
    const_view5 = np.zeros((height-4,width-4))

    
    # calculating Disparity Maps for View1
    for i in range(5,height-4):
        for j in range(5,width-4):
            view1_slice = view1_padded[i-4:i+5,j-4:j+5]
            min_list = []
            
            for k in range(j-75,j):
                if (k < 5 ):
                    k=5
                view5_slice = view5_padded[i-4:i+5,k-4:k+5]
                view1_diff = np.subtract(view1_slice,view5_slice)
                view1_square = np.multiply(view1_diff,view1_diff)
                view1_sum = np.sum(view1_square)
                min_list.append([view1_sum,k])
                #print i,j,k            
            min_list.sort()
            temp = min_list[0]
            dmap1[i][j] = j-temp[1]
    
    dmap1_img = dmap1/dmap1.max()
    cv2.imshow("DMAP1_9x9",dmap1_img)
    
    # calculating Disparity Maps for View5
    for i in range(5,height-4):
        for j in range(5,width-4):
            view5_slice = view5_padded[i-4:i+5,j-4:j+5]
            min_list = []
            for k in range(j+100,j,-1):
                if (k >= width-5 ):
                    k=width-6
                view1_slice = view1_padded[i-4:i+5,k-4:k+5]
                #print i,j,k
                #print view1_slice
                view5_diff = np.subtract(view1_slice,view5_slice)
                view5_square = np.multiply(view5_diff,view5_diff)
                view5_sum = np.sum(view5_square)
                min_list.append([view5_sum,k])
                
            min_list.sort()
            temp = min_list[0]
            dmap2[i][j] = temp[1]-j
    
    dmap2_img = dmap2/dmap2.max()
    cv2.imshow("DMAP2_9x9",dmap2_img)
    
    dmap1 = dmap1[4:height-4,4:width-4]
    dmap2 = dmap2[4:height-4,4:width-4]
    
    
    #### MSE calculation loop ###
    # view1
    mse_diff = np.subtract(image_gt1,dmap1)
    mse_square = np.multiply(mse_diff,mse_diff)
    mse_square_sum = np.sum(mse_square)
    mse_view1 = mse_square_sum/(height*width)
    print "9x9:MSE value for View1 is:",mse_view1
    
    # view5
    mse_diff = np.subtract(image_gt5,dmap2)
    mse_square = np.multiply(mse_diff,mse_diff)
    mse_square_sum = np.sum(mse_square)
    mse_view5 = mse_square_sum/(height*width)
    print "9x9:MSE value for View5 is:",mse_view5

    height_d, width_d = dmap1.shape
    print "Dmap height width", height_d,width_d
    
    #setting 0's in ground truth
    list_zeros = np.where(dmap1==0)
    list_zeros_length = len(list_zeros)
    for i in range(0,list_zeros_length):
        x=list_zeros[0][i]
        y=list_zeros[1][i]
        image_gt1[x][y]=0
    
    list_zeros = np.where(dmap2==0)
    list_zeros_length = len(list_zeros)
    for i in range(0,list_zeros_length):
        x=list_zeros[0][i]
        y=list_zeros[1][i]
        image_gt5[x][y]=0


    for i in range(0,height_d):
        for j in range(0,width_d):
            disparity = dmap1[i][j]
            temp = np.absolute(j-disparity)
            if (dmap2[i][temp] != disparity):
                dmap2[i][temp]=0
                dmap1[i][j]=0   
    
    for i in range(0, height_d):
        for j in range(0,width_d):
            disparity = dmap2[i][j]
            temp = np.absolute(j+disparity)
            if (dmap1[i][temp] != disparity):
                dmap1[i][temp]=0
                dmap2[i][j]=0   
            
                
    
    for i in range(0, height_d):
        for j in range(0, width_d):
            if (dmap1[i][j] == 0):
                image_gt1[i][j] = 0
            if (dmap2[i][j] == 0):
                image_gt5[i][j] = 0    
    
    ### MSE calculation loop ###
    # view1`
    mse_diff = np.subtract(image_gt1,dmap1)
    mse_square = np.multiply(mse_diff,mse_diff)
    mse_square_sum = np.sum(mse_square)
    mse_view1 = mse_square_sum/(height*width)
    print "9x9:MSE value for View1 after Consistency check is:",mse_view1
    
    # view5
    mse_diff = np.subtract(image_gt5,dmap2)
    mse_square = np.multiply(mse_diff,mse_diff)
    mse_square_sum = np.sum(mse_square)
    mse_view5 = mse_square_sum/(height*width)
    print "9x9:MSE value for View5 after Consistency check is:",mse_view5
    
    
    const_view1 = dmap1/dmap1.max()
    const_view5 = dmap2/dmap2.max()
    
    cv2.imshow("Consitency_Map1_9x9",const_view1)
    cv2.imshow("Consitency_Map2_9x9",const_view5)


threeblock()
nineblock()