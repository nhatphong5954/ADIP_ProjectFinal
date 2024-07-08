import cv2 as cv
import numpy as np
import operatorADIP

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

#operatorADIP.hit_or_miss_and_display(input_image, kernel)
#operatorADIP.binary_region_filling(input_image, (1, 1))
#operatorADIP.grayscale_opening(input_image, kernel)
#operatorADIP.grayscale_morphology_gradient(input_image, kernel)
#operatorADIP.binary_opening(input_image, kernel)
#operatorADIP.binary_thinning(input_image)
#operatorADIP.grayscale_top_hat_transformation(input_image, kernel)
