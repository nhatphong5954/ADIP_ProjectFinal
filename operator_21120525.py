import cv2 as cv
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import Tk
#2120525 - Cao Nhật Phong
#Cài đặt các toán tử sau

def hit_or_miss(input_image, kernel, rate=50):
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

def binary_region_filling(input_image, seed_point, rate=50):
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
    
    input_image_display = cv.resize(input_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Original", input_image_display)
    cv.moveWindow("Original", 0, 200)
    
    output_image_display = cv.resize(output_image, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    cv.imshow("Binary Region Filling", output_image_display)
    cv.moveWindow("Binary Region Filling", 500, 200)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    
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

def menu():
    print("--------------------- Menu ---------------------")
    print("1. Hit or Miss")
    print("2. Binary Region Filling")
    print("3. Grayscale Opening")
    print("4. Grayscale Morphology Gradient")
    print("5. Binary Opening")
    print("6. Binary Thinning")
    print("7. Grayscale Top-hat Transformation")
    print("0. Exit")
    choice = input("Your choice: ")
    return choice


def main():

    # Ẩn cửa sổ do hàm Tk() gọi
    Tk().withdraw()

    # Người dùng chọn ảnh trên máy
    image_path = askopenfilename(title="Select a grayscale image",
                                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])

    # Kiểm tra file có tồn tại không
    if not image_path:
        raise FileNotFoundError("No file selected. Please select an image file.")

    # Tải ảnh mức xám lên, nếu là ảnh màu thì không thông qua hàm này
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Kiểm tra ảnh tải lên được không
    if image is None:
        raise FileNotFoundError(f"The image could not be loaded. Please check the file and try again.")
    
    choice = -1
    while choice != 0:

        choice = menu()
        if choice == "0":
            print("Exit!")
            exit()
            
        
        if choice == "1":
            ImageOutput = hit_or_miss(image)

        elif choice == "2":
            ImageOutput = binary_region_filling(image)
            
        elif choice == "3":
            ImageOutput = grayscale_opening(image_path)
        
        elif choice == "4":
            ImageOutput = grayscale_morphology_gradient(image)
            
        elif choice == "5":
            ImageOutput = binary_opening(image)

        elif choice == "6":
            ImageOutput = binary_thinning(image)

        elif choice == "7":
            ImageOutput = grayscale_top_hat_transformation(image_path)
        
        else:
            print("Invalid choice. Please choose again.")
            exit()

if __name__ == "__main__":
    main()