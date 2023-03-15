import cv2
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

def ssim_calc(im1, im2):
    return ssim(im1, im2, channel_axis=2)

kernels = []
for theta in range(3):
    theta = theta / 3. * np.pi
    for sigma in (1, 3, 6):
        for frequency in (0.05, 0.10, 0.15):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                            sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


def compute_feats(image):
    global kernels
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        # feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def gabore_compare(im1, im2):
    global kernels, prev_hash, cache

    im1 = cv2.resize(im1, (256, 256))
    im2 = cv2.resize(im2, (256, 256))

    a1 = compute_feats(cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), (128, 128)))
    a2 = compute_feats(cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), (128, 128)))

    part2 = np.sum(np.abs(a1[:,1] - a2[:,1]))
    return part2


def gabor(image):
    real, _ = skimage.filters.gabor(
        image, frequency=0.15, theta=np.pi / 3, sigma_x=3, sigma_y=3, mode="wrap"
    )

    return np.array(cv2.meanStdDev(real))


def gabor_calc(im1, im2):
    """ktau=0.71

    * x4 Downsample does not change quality (-0.006)
    * frequencies (0.05, 0.10, 0.15) to 0.10 decreases quality (-0.02)
    """

    gabor_1 = gabor(cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), (128, 128)))
    gabor_2 = gabor(cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), (128, 128)))

    return -np.linalg.norm(gabor_1 - gabor_2)


def sobel(image):
    grad_x = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=13)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=13)

    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)

    return grad


def sobel_calc(im1, im2):
    """Calculates norm of image edge difference

    * Large kernel size increases quality (+0.05)
    * Grayscale conversion descreases quality (-0.02)
    * Histogram of edges descreases quality (-0.04)
    * x4 Downsample decreases quality (-0.05)
    * SSIM instead of norm decreases quality (-0.12)
    """

    edge_1 = sobel(im1)
    edge_2 = sobel(im2)

    return np.linalg.norm(edge_1 - edge_2)


hog = cv2.HOGDescriptor()


def hog_calc(im1, im2):
    """Calculates norm of image hog descriptors difference

    * Grayscale conversion increases quality (+0.05)
    * x4 Downsample increases quality (+0.04)
    """

    im1 = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), (128, 128))
    im2 = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), (128, 128))

    hog_1 = hog.compute(im1)
    hog_2 = hog.compute(im2)

    return np.linalg.norm(hog_1 - hog_2)


def lbp(image):
    edges = np.rint(sobel(image)).astype(np.uint8)
    gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    patterns = skimage.feature.local_binary_pattern(gray, P=4, R=8)

    return patterns


def lbp_calc(im1, im2):
    """Calculates norm of image lbp descriptors difference

    * P and R tuning increases quality (+0.11)
    * Edge detectcion increases quality (+0.09)
    * Histogram calculation decreases quality (-0.06)
    """

    lbp_1 = lbp(im1)
    lbp_2 = lbp(im2)

    return np.linalg.norm(lbp_1 - lbp_2)


def haff_calc(im1, im2):
    """Calculates norm of image lines difference

    * Canny threshold tuning increases quality
    * Line thickness increases quality
    """

    edges1 = cv2.Canny(im1, 150, 255)
    edges2 = cv2.Canny(im2, 150, 255)
    lines1 = cv2.HoughLinesP(edges1, 200, np.pi / 3, 150, None, 0, 0)
    lines2 = cv2.HoughLinesP(edges2, 200, np.pi / 3, 150, None, 0, 0)
    image1 = np.zeros_like(im1)
    image2 = np.zeros_like(im2)
    if lines1 is not None:
        for linee in lines1:
            line = linee[0]
            cv2.line(image1, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), thickness=5)
    if lines2 is not None:
        for linee in lines2:
            line = linee[0]
            cv2.line(image2, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), thickness=5)

    return np.linalg.norm(image1 - image2)
