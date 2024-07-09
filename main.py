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
    oper525.hit_or_miss_and_display(input_image, kernel)
    #oper525.binary_region_filling(input_image, (1, 1))
    #oper525.grayscale_opening(input_image, kernel)
    #oper525.grayscale_morphology_gradient(input_image, kernel)
    #oper525.binary_opening(input_image, kernel)
    #oper525.binary_thinning(input_image)
    #oper525.grayscale_top_hat_transformation(input_image, kernel)

    # # Ẩn cửa sổ do hàm Tk() gọi
    # Tk().withdraw()

    # # Người dùng chọn ảnh trên máy
    # image_path = oper029.askopenfilename(title="Select a grayscale image",
    #                             filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])

    # # Kiểm tra file có tồn tại không
    # if not image_path:
    #     raise FileNotFoundError("No file selected. Please select an image file.")

    # # Tải ảnh mức xám lên, nếu là ảnh màu thì không thông qua hàm này
    # image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # # Kiểm tra ảnh tải lên được không
    # if image is None:
    #     raise FileNotFoundError(f"The image could not be loaded. Please check the file and try again.")
    
    # choice = -1
    # while choice != 0:

    #     choice = oper029.menu()
    #     if choice == "0":
    #         print("Exit!")
    #         exit()
            
        
    #     if choice == "1":
    #         ImageOutput = oper029.grayscale_smoothing(image)

    #     elif choice == "2":
    #         ImageOutput = oper029.grayscale_closing(image)
            
    #     elif choice == "3":
    #         ImageOutput = oper029.binary_closing(image_path)
        
    #     elif choice == "4":
    #         ImageOutput = oper029.grayscale_reconstruction(image)
            
    #     elif choice == "5":
    #         ImageOutput = oper029.grayscale_textural_segmentation(image)

    #     elif choice == "6":
    #         ImageOutput = oper029.grayscale_granulometry(image)

    #     elif choice == "7":
    #         ImageOutput = oper029.binary_convex_hull(image_path)
        
    #     else:
    #         print("Invalid choice. Please choose again.")
    #         exit()

if __name__ == "__main__":
    main()