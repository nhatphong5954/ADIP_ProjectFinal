import cv2 as cv
import numpy as np

#Nhat Phong: 

def hit_or_miss_and_display(input_image, kernel, rate=50):
    output_image = cv.morphologyEx(input_image, cv.MORPH_HITMISS, kernel)
    
    kernel_display = (kernel + 1) * 127
    kernel_display = np.uint8(kernel_display)
    kernel_display = cv.resize(kernel_display, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("kernel", kernel_display)
    cv.moveWindow("kernel", 0, 0)
    
    input_image_display = cv.resize(input_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Original", input_image_display)
    cv.moveWindow("Original", 0, 200)
    
    output_image_display = cv.resize(output_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Hit or Miss", output_image_display)
    cv.moveWindow("Hit or Miss", 500, 200)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

def binary_region_filling(input_image, seed_point):
    h, w = input_image.shape
    output_image = np.zeros((h, w), np.uint8)
    output_image[seed_point] = 255
    prev_image = output_image.copy()
    
    while True:
        dilated_image = cv.dilate(output_image, None, iterations=1)
        output_image = cv.bitwise_and(dilated_image, input_image)
        if np.array_equal(prev_image, output_image):
            break
        prev_image = output_image.copy()
    
    return output_image

def grayscale_opening(input_image, kernel, rate=50):
    kernel = np.uint8(kernel)
    output_image = cv.morphologyEx(input_image, cv.MORPH_OPEN, kernel)
    
    kernel_display = (kernel + 1) * 127
    kernel_display = np.uint8(kernel_display)
    kernel_display = cv.resize(kernel_display, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("kernel", kernel_display)
    cv.moveWindow("kernel", 0, 0)
    
    input_image_display = cv.resize(input_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Original", input_image_display)
    cv.moveWindow("Original", 0, 200)
    
    output_image_display = cv.resize(output_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Grayscale Opening", output_image_display)
    cv.moveWindow("Grayscale Opening", 500, 200)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

def grayscale_morphology_gradient(input_image, kernel, rate=50):
    kernel = np.uint8(kernel)
    output_image = cv.morphologyEx(input_image, cv.MORPH_GRADIENT, kernel)
    
    kernel_display = (kernel + 1) * 127
    kernel_display = np.uint8(kernel_display)
    kernel_display = cv.resize(kernel_display, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("kernel", kernel_display)
    cv.moveWindow("kernel", 0, 0)
    
    input_image_display = cv.resize(input_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Original", input_image_display)
    cv.moveWindow("Original", 0, 200)
    
    output_image_display = cv.resize(output_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Grayscale Morphology Gradient", output_image_display)
    cv.moveWindow("Grayscale Morphology Gradient", 500, 200)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

def binary_opening(input_image, kernel, rate=50):
    kernel = np.uint8(kernel)
    output_image = cv.morphologyEx(input_image, cv.MORPH_OPEN, kernel)
    
    kernel_display = (kernel + 1) * 127
    kernel_display = np.uint8(kernel_display)
    kernel_display = cv.resize(kernel_display, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("kernel", kernel_display)
    cv.moveWindow("kernel", 0, 0)
    
    input_image_display = cv.resize(input_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Original", input_image_display)
    cv.moveWindow("Original", 0, 200)
    
    output_image_display = cv.resize(output_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Binary Opening", output_image_display)
    cv.moveWindow("Binary Opening", 500, 200)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

def binary_thinning(input_image, rate=50):
    size = np.size(input_image)
    skeleton = np.zeros(input_image.shape, np.uint8)
    
    ret, input_image = cv.threshold(input_image, 127, 255, cv.THRESH_BINARY)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False
    
    while not done:
        eroded = cv.erode(input_image, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(input_image, temp)
        skeleton = cv.bitwise_or(skeleton, temp)
        input_image = eroded.copy()
        
        zeros = size - cv.countNonZero(input_image)
        if zeros == size:
            done = True

    input_image_display = cv.resize(input_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Original", input_image_display)
    cv.moveWindow("Original", 0, 200)
    
    skeleton_display = cv.resize(skeleton, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Binary Thinning", skeleton_display)
    cv.moveWindow("Binary Thinning", 500, 200)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

def grayscale_top_hat_transformation(input_image, kernel, rate=50):
    kernel = np.uint8(kernel)
    output_image = cv.morphologyEx(input_image, cv.MORPH_TOPHAT, kernel)
    
    kernel_display = (kernel + 1) * 127
    kernel_display = np.uint8(kernel_display)
    kernel_display = cv.resize(kernel_display, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("kernel", kernel_display)
    cv.moveWindow("kernel", 0, 0)
    
    input_image_display = cv.resize(input_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Original", input_image_display)
    cv.moveWindow("Original", 0, 200)
    
    output_image_display = cv.resize(output_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Grayscale Top-hat Transformation", output_image_display)
    cv.moveWindow("Grayscale Top-hat Transformation", 500, 200)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
