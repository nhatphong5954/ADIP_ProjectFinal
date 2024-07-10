import cv2 as cv
import numpy as np
import operator_21120525 as oper525
import operator_20120029 as oper029
input_image = np.array((
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 255, 255, 255, 0, 0, 0, 255],
 [0, 255, 255, 255, 0, 0, 0, 0],
 [0, 255, 255, 255, 0, 255, 0, 0],
 [0, 0, 255, 0, 0, 0, 0, 0],
 [0, 0, 255, 0, 0, 255, 255, 0],
 [0, 255, 0, 255, 0, 0, 255, 0],
 [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")
 
kernel = np.array((
 [0, 1, 0],
 [1, -1, 1],
 [0, 1, 0]), dtype="int")

def main():
    oper525.hit_or_miss(input_image, kernel)
    #oper525.binary_region_filling(input_image, (1, 1))
    #oper525.grayscale_opening(input_image, kernel)
    #oper525.grayscale_morphology_gradient(input_image, kernel)
    #oper525.binary_opening(input_image, kernel)
    #oper525.binary_thinning(input_image)
    #oper525.grayscale_top_hat_transformation(input_image, kernel)

if __name__ == "__main__":
    main()