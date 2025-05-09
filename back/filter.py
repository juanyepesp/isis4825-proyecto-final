import cv2
import numpy as np
from skimage.segmentation import chan_vese
from skimage.morphology import dilation, erosion
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

def process_image_array(img_array: np.ndarray) -> np.ndarray:
    img_rgb = img_array[:, :, :3]

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

    den_gauss = gaussian(morph, sigma=1.5, preserve_range=True).astype(np.uint8)
    den_med = median(den_gauss, np.ones((5,5), np.uint8))
    if den_med.dtype != np.uint8:
        den_med = (den_med * 255).astype(np.uint8) if den_med.max() <= 1.0 else den_med.astype(np.uint8)
    guided = guidedFilter(den_med, gray, radius=5, eps=1e-2)
    R = guided

    gabor_feats = apply_gabor_bank(den_med)
    gabor_combined = np.mean(gabor_feats, axis=0).astype(np.uint8)
    G = cv2.normalize(gabor_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    segments = chan_vese(gray, mu=0.25, tol=1e-3, max_num_iter=100, dt=0.5, init_level_set="checkerboard", extended_output=True)
    img_segments = np.zeros_like(gray)
    for i, label in enumerate(np.unique(segments[0])):
        mask = (segments[0] == label)
        img_segments[mask] = (i + 1) * (255 // len(np.unique(segments[0])))
    B = cv2.normalize(img_segments, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    rgb_image = np.dstack((R, G, B))
    return rgb_image

