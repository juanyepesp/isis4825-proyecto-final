import os
import sys
import cv2
import gc
import numpy as np
from skimage import io
from skimage.segmentation import chan_vese
from skimage.morphology import dilation, erosion
from skimage.segmentation import slic
from skimage.filters.rank import median
from skimage.filters import gaussian
from cv2.ximgproc import guidedFilter

def mask_overlays(gray):
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def apply_gabor_bank(img, ksize=15, sig=2.0, lam=10.0, gamma=0.5):
    thetas = np.arange(0, np.pi, np.pi / 6)
    responses = []
    for theta in thetas:
        kern = cv2.getGaborKernel((ksize, ksize), sig, theta, lam, gamma)
        fimg = cv2.filter2D(img, cv2.CV_32F, kern)
        fimg = cv2.normalize(fimg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        responses.append(fimg)
    return responses

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_rgb = img[:, :, :3]
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    gray = mask_overlays(gray)
    
    eroded_1 = erosion(gray, footprint=np.ones((5,5), dtype=np.uint8))
    dilated_1 = dilation(eroded_1, footprint=np.ones((5,5), dtype=np.uint8))
    cross_footprint = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.uint8)
    dilated_2 = dilation(dilated_1, footprint=cross_footprint)
    morph = np.minimum(dilated_2, gray)

    # R channel: Gaussian + Median + Guided Filter
    den_gauss = gaussian(morph, sigma=1.5, preserve_range=True)
    den_gauss = (den_gauss).astype(np.uint8)  
    den_med = median(den_gauss, np.ones((5,5), np.uint8))
    if den_med.dtype != np.uint8:
        den_med = (den_med * 255).astype(np.uint8) if den_med.max() <= 1.0 else den_med.astype(np.uint8)
    guided = guidedFilter(den_med, gray, radius=5, eps=1e-2)
    R = guided

    # G channel: Gabor filter
    gabor_feats = apply_gabor_bank(den_med)
    gabor_combined = np.mean(gabor_feats, axis=0).astype(np.uint8)
    G = cv2.normalize(gabor_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # B channel: Segmentation Chan-Vese
    segments = chan_vese(gray, mu=0.25, tol=1e-3, max_num_iter=100, dt=0.5, init_level_set="checkerboard", extended_output=True)
    img_segments = np.zeros_like(gray)
    for i, label in enumerate(np.unique(segments[0])):
        mask = (segments[0] == label)
        img_segments[mask] = (i + 1) * (255 // len(np.unique(segments[0])))
    B = cv2.normalize(img_segments, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Combine channels
    rgb_image = np.dstack((R,G,B))
    del gray, R, G, B, img_segments, gabor_feats, gabor_combined, segments
    return rgb_image

def main(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        file_index = 0
        while file_index < len(files):
            file = files[file_index]
            file_index += 1
            in_path = os.path.join(root, file)
            rel = os.path.relpath(in_path, input_dir)
            out_dir = os.path.join(output_dir, os.path.dirname(rel))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, file)
            try:
                rgb_img = process_image(in_path)
            except Exception as e:
                print(f"Failed: {in_path} ({type(e).__name__}: {e})")
                continue

            base, _ = os.path.splitext(out_path)
            cv2.imwrite(base + '.jpg', rgb_img)

            print(f"Processed: IN {in_path} -> OUT {out_path}")
            del rgb_img
            gc.collect()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter.py <input_folder_path> <output_folder_path>")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    if not os.path.exists(inp):
        print(f"Input folder not found: {inp}")
        sys.exit(1)
    main(inp, outp)
