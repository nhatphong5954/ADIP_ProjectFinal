# Lê Thị Minh Thư - 21120184

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Erosion nhị phân
def binary_erosion(image_path):
    
    # Tải ảnh màu lên
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Đổi từ ảnh màu sang ảnh mức xám
    binary_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Kiểm tra ảnh màu
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    
    try:
        kernel_size = int(input("Enter the kernel size (positive odd number, ex: 3 for a 3x3 kernel): "))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number.")
    except ValueError as e:
        raise ValueError(f"Invalid kernel size: {e}")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Tính toán toán tử erosion
    erosion = cv2.erode(binary_image, kernel, iterations=1)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(binary_image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(erosion, cmap='gray'), plt.title('Binary Erosion'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Thicken nhị phân
def binary_thickening(image_path):

    # Tải ảnh màu lên
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Đổi từ ảnh màu sang ảnh mức xám
    binary_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Kiểm tra ảnh màu
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    
    try:
        kernel_size = int(input("Enter the kernel size (positive odd number, ex: 3 for a 3x3 kernel): "))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number.")
    except ValueError as e:
        raise ValueError(f"Invalid kernel size: {e}")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Tính toán toán tử thickening
    thickening = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(binary_image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(thickening, cmap='gray'), plt.title('Binary Thickening'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Erosion mức xám
def grayscale_erosion(image):

    try:
        kernel_size = int(input("Enter the kernel size (positive odd number, ex: 3 for a 3x3 kernel): "))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number.")
    except ValueError as e:
        raise ValueError(f"Invalid kernel size: {e}")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Tính toán toán tử erosion mức xám
    erosion = cv2.erode(image, kernel, iterations=1)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(erosion, cmap='gray'), plt.title('Grayscale Erosion'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Dilation mức xám
def grayscale_dilation(image):

    try:
        kernel_size = int(input("Enter the kernel size (positive odd number, ex: 3 for a 3x3 kernel): "))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number.")
    except ValueError as e:
        raise ValueError(f"Invalid kernel size: {e}")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Tính toán toán tử dilation mức xám
    dilation = cv2.dilate(image, kernel, iterations=1)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dilation, cmap='gray'), plt.title('Grayscale Dilation'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Skeleton nhị phân
def binary_skeleton(image_path):

    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ ảnh
    blur = cv2.blur(gray_image, (3, 3))
    
    # Ảnh threshold
    _, binary_image = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    
    # Tìm skeleton cho ảnh
    size = np.size(binary_image)
    skeleton = np.zeros(binary_image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        open = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(binary_image, open)
        eroded = cv2.erode(binary_image, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_image = eroded.copy()
        if cv2.countNonZero(binary_image) == 0:
            break

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(binary_image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(skeleton, cmap='gray'), plt.title('Binary Skeleton'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Dilation nhị phân
def binary_dilation(image_path):

    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ ảnh
    blur = cv2.blur(gray_image, (3, 3))
    
    # Ảnh threshold
    _, binary_image = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    
    try:
        kernel_size = int(input("Enter the kernel size (positive odd number, ex: 3 for a 3x3 kernel): "))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number.")
    except ValueError as e:
        raise ValueError(f"Invalid kernel size: {e}")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Tính toán toán tử dilation nhị phân
    dilation = cv2.dilate(binary_image, kernel, iterations=1)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(binary_image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dilation, cmap='gray'), plt.title('Binary Dilation'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Phân ranh giới nhị phân
def binary_boundary_extraction(image_path):

    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ ảnh
    blur = cv2.blur(gray_image, (3, 3))
    
    # Ảnh threshold
    _, binary_image = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    
    try:
        kernel_size = int(input("Enter the kernel size (positive odd number, ex: 3 for a 3x3 kernel): "))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number.")
    except ValueError as e:
        raise ValueError(f"Invalid kernel size: {e}")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Tính toán toán tử erosion nhị phân
    erosion = cv2.erode(binary_image, kernel, iterations=1)

    # Tính toán ảnh phân ranh giới
    boundary = binary_image - erosion

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(binary_image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(boundary, cmap='gray'), plt.title('Binary Boundary Extraction'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

def menu():
    print("--------------------- Menu ---------------------")
    print("1. Binary Erosion")
    print("2. Binary Thickening")
    print("3. Grayscale Erosion")
    print("4. Grayscale Dilation")
    print("5. Binary Skeleton")
    print("6. Binary Dilation")
    print("7. Binary Boundary Extraction")
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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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
            ImageOutput = binary_erosion(image_path)

        elif choice == "2":
            ImageOutput = binary_thickening(image_path)

        elif choice == "3":
            ImageOutput = grayscale_erosion(image)

        elif choice == "4":
            ImageOutput = grayscale_dilation(image)

        elif choice == "5":
            ImageOutput = binary_skeleton(image_path)

        elif choice == "6":
            ImageOutput = binary_dilation(image_path)

        elif choice == "7":
            ImageOutput = binary_boundary_extraction(image_path)
        
        else:
            print("Invalid choice. Please choose again.")
            exit()

if __name__ == "__main__":
    main()
