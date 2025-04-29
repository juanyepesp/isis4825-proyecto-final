import os
import sys
import cv2
import numpy as np
from skimage import io
from skimage.exposure import equalize_adapthist

def process_image(image_path):
    img = io.imread(image_path)
    img_resized = cv2.resize(img, (300, 300))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    img_equalized = equalize_adapthist(img_blur)
    img_equalized = (img_equalized * 255).astype(np.uint8)

    sobelx = cv2.Sobel(img_equalized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_equalized, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    laplacian = cv2.Laplacian(img_equalized, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

    feature_image = np.dstack((img_equalized, sobel_mag, laplacian))
    return feature_image

def main(input_dir, output_dir):
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(supported_ext):
                input_path = os.path.join(root, file)

                rel_path = os.path.relpath(input_path, input_dir)
                label = os.path.dirname(rel_path)

                output_label_dir = os.path.join(output_dir, label)
                os.makedirs(output_label_dir, exist_ok=True)

                output_path = os.path.join(output_label_dir, file)

                try:
                    processed_img = process_image(input_path)
                    io.imsave(output_path, processed_img)
                    print(f"Processed: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        print("Example: python filter.py data/Imagenes/train data/Procesado/train")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        sys.exit(1)

    main(input_folder, output_folder)
