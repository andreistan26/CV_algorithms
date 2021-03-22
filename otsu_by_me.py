import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize) #used for uncapping the max stdout in terminal

need_noise_reduction = True

def main():
    picture = cv2.imread("../../../res/lenna.png", 0);
    if need_noise_reduction:
        picture = cv2.GaussianBlur(picture, (5, 5), 0)
    #hist
    bin_num = 256
    hist,_ = np.histogram(picture, bins=bin_num)
    #P(hist)
    prob_dist_arr = hist/np.sum(hist)
    max_val = 1e+10
    thr = 0
    for i in range(1, 255):
        ar1 = prob_dist_arr[:i]
        ar2 = prob_dist_arr[i+1:]
        q1 = np.sum(ar1)
        q2 = 1-q1
        m1 = np.sum(np.array([j for j in range(i)])*prob_dist_arr[:i])/q1
        m2 = np.sum(np.array([j for j in range(i,256)])*prob_dist_arr[i:])/q2
        o1 = np.sum(np.array([np.square(j-m1)*prob_dist_arr[j]/q1 for j in range(i)]))                       
        o2 = np.sum(np.array([np.square(j-m2)*prob_dist_arr[j]/q2 for j in range(i,256)]))                   
        weight = q1*o1+q2*o2                                                                                 
        if max_val > weight:
            max_val = weight
            thr = i

    #opencv threshold
    otsu_t,_ = cv2.threshold(picture, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #matplotlib plots
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].hist(picture.ravel(),bins=256, range=(0,255))
    ax[0, 0].set_title('image histogram')
    ax[0, 1].imshow(picture)
    ax[0, 1].set_title('original picture')
    ax[1, 0].imshow(picture > thr, cmap='gray')
    ax[1, 0].set_title(f'our otsu implementation; t={thr}')
    ax[1, 1].imshow(picture > otsu_t, cmap='gray')
    ax[1, 1].set_title(f'otsu threshold by opencv; t={otsu_t}')
    plt.show()

if __name__=='__main__':
    main()
