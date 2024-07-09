# 20120029 - Nguyễn Minh An
# Cài đặt các hàm: làm trơn, đóng (mức xám & nhị phân), hồi phục, phân đoạn vân, đếm hạt, bao lồi

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Toán tử làm trơn
def grayscale_smoothing(image):
    
    # Người dùng chọn kernel
    try:
        kernel_size = int(input("Enter the kernel size (positive odd number, ex: 3 for a 3x3 kernel): "))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number.")
    except ValueError as e:
        raise ValueError(f"Invalid kernel size: {e}")
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    smoothing = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Tiến trình tính toán
    dilation = cv2.dilate(smoothing, kernel, iterations=1)
    erosion = cv2.erode(smoothing, kernel, iterations=1)
    opening = cv2.morphologyEx(smoothing, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(smoothing, cv2.MORPH_CLOSE, kernel)

    # Kết quả
    plt.figure(figsize=(5, 4))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(smoothing, cmap='gray'), plt.title('Grayscale Smoothing'), plt.xticks([]), plt.yticks([])
    plt.show()

# Toán tử đóng độ xám    
def grayscale_closing(image):
    
    try:
        kernel_size = int(input("Enter the kernel size (positive odd number, ex: 3 for a 3x3 kernel): "))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number.")
    except ValueError as e:
        raise ValueError(f"Invalid kernel size: {e}")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Tính toán toán tử đóng
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(closing, cmap='gray'), plt.title('Grayscale Closing'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Toán tử đóng nhị phân
def binary_closing(image_path):
    
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

    # Tính toán toán tử đóng
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(binary_image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(closing, cmap='gray'), plt.title('Binary Closing'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Toán tử hồi phục
def grayscale_reconstruction(image):
    
    kernel = np.ones((5, 5), np.uint8)
    marker = cv2.erode(image, kernel, iterations=1)
    
    # Tái tạo ảnh qua phép giãn nở
    reconstructed = cv2.dilate(marker, kernel, iterations=1)
    for i in range(10):
        prev_reconstructed = reconstructed
        reconstructed = cv2.min(image, cv2.dilate(reconstructed, kernel, iterations=1))
        if np.array_equal(prev_reconstructed, reconstructed):
            break

    plt.figure(figsize=(12, 8))
    plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(marker, cmap='gray'), plt.title('Marker Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(reconstructed, cmap='gray'), plt.title('Grayscale Reconstruction'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    
# Toán tử phân đoạn vân    
def grayscale_textural_segmentation(image):
    
    try:
        kernel_size = int(input("Enter the kernel size (positive odd number, ex: 3 for a 3x3 kernel): "))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd number.")
    except ValueError as e:
        raise ValueError(f"Invalid kernel size: {e}")
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Tiến trình tính toán
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    text_seg = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    plt.figure(figsize=(12, 8))
    plt.subplot(241), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(242), plt.imshow(text_seg, cmap='gray'), plt.title('Textural Segmentation'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# Toán tử đếm hạt
def grayscale_granulometry(image):
    
    # Khởi tạo mảng để lưu granulometry 
    sizes = []
    granulometry = []

    # Biểu diễn granulometry bằng toán tử mở độ xám với kích thước phần tử tăng dần
    for size in range(1, 21, 2):  # Tăng kích thước phần tử
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Tính toán chênh lệch pixel trước và sau khi mở
        diff = cv2.absdiff(image, opened_image)
        
        # Tính tổng lại
        structure_removed = np.sum(diff)
        
        # Lưu kích thước với granulometry tương ứng
        sizes.append(size)
        granulometry.append(structure_removed)

    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')

    plt.subplot(122)
    plt.plot(sizes, granulometry, marker='o')
    plt.title('Grayscale Granulometry')
    plt.xlabel('Structuring Element Size')
    plt.ylabel('Amount of Structure Removed')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Bao lồi    
def binary_convex_hull(image_path):

    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ ảnh
    blur = cv2.blur(gray_image, (3, 3))
    
    # Ảnh threshold
    _, binary_image = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    
    # Tìm đường viền ảnh
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tìm bao lồi cho viền ảnh
    hull = [cv2.convexHull(contour) for contour in contours]
    
    # Tạo ảnh mới để vẽ bao lồi
    hull_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), np.uint8)
    
    # Vẽ viền và bao lồi
    for i in range(len(contours)):
        cv2.drawContours(hull_image, contours, i, (0, 255, 0), 1, 8)
        cv2.drawContours(hull_image, hull, i, (255, 0, 0), 1, 8)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(binary_image, cmap='gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(hull_image, cmap='gray'), plt.title('Binary Convex Hull'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    

def menu():
    print("--------------------- Menu ---------------------")
    print("1. Grayscale Smoothing")
    print("2. Grayscale Closing")
    print("3. Binary Closing")
    print("4. Grayscale Reconstruction")
    print("5. Grayscale Textural Segmentation")
    print("6. Grayscale Granulometry")
    print("7. Binary Convex Hull")
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
            ImageOutput = grayscale_smoothing(image)

        elif choice == "2":
            ImageOutput = grayscale_closing(image)
            
        elif choice == "3":
            ImageOutput = binary_closing(image_path)
        
        elif choice == "4":
            ImageOutput = grayscale_reconstruction(image)
            
        elif choice == "5":
            ImageOutput = grayscale_textural_segmentation(image)

        elif choice == "6":
            ImageOutput = grayscale_granulometry(image)

        elif choice == "7":
            ImageOutput = binary_convex_hull(image_path)
        
        else:
            print("Invalid choice. Please choose again.")
            exit()

if __name__ == "__main__":
    main()