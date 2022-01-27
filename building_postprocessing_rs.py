import cv2 as cv
import numpy as np
from skimage.util import img_as_ubyte
from skimage.morphology import skeletonize
from scipy import ndimage


def thick_edge2one_pixel(edge_img, th=150):
    """
    Thin thick edges.
    :param edge_img: an image that is predicted by deep learning edge models, opencv mat
    :param th: threshold
    :return:  an image with 1 pixel wide edges
    """
    img_f = edge_img.flatten()
    img_bin_1d = np.where(img_f <= th, 0, 1)
    img_bin = img_bin_1d.reshape(edge_img.shape)
    img_bin = img_as_ubyte(img_bin)
    skeleton = skeletonize(img_bin)
    img1pix = img_as_ubyte(skeleton)
    return img1pix


def add_edge2poly(edge_img, poly_img):
    """
    Combine the edge and region image
    :param edge_img: binary edge image
    :param poly_img: binary region image
    :return: logical OR of edges and regions
    """
    img = edge_img + poly_img
    img_f = img.flatten()
    img_bin_1d = np.where(img_f < 1, 0, 255)
    img_bin = img_bin_1d.reshape(edge_img.shape)
    return img_bin


def refine_edge2poly(edgeorpoly):
    """
    Fill holes in the image and remove redundant edges.
    :param edgeorpoly: combination of edges and regions
    :return: refined image
    """
    # step 1
    img_fill_hole = ndimage.binary_fill_holes(edgeorpoly)
    # step 2
    kernel = np.ones((3, 3), np.uint8)
    img_fill_hole = img_as_ubyte(img_fill_hole)
    out_erosion = cv.erode(img_fill_hole, kernel, iterations=1)
    out_erosion_dif = img_fill_hole - out_erosion
    return out_erosion, out_erosion_dif


def remove_small_regions(img, th=16):
    """
    Remove regions smaller that the specified area.
    :param img: input image
    :param th: area threshold
    :return: image
    """
    if th == 0:
        return img

    w, h = img.shape
    poly_ret, poly_marker = cv.connectedComponents(img, connectivity=8)
    poly_del = []
    poly_contours, poly_hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in poly_contours:
        cnt_area = cv.contourArea(cnt)
        if cnt_area <= th:
            cn = poly_marker[cnt[0, 0, 1], cnt[0, 0, 0]]
            poly_del.append(cn)

    for i in range(0, w):
        for j in range(0, h):
            if poly_marker[i, j] in poly_del:
                img[i, j] = 0
    return img


def building_ext_post(poly_path, edge_path, save_path, save_edge_path, edge_th, area_th):
    poly_img = cv.imread(poly_path, 0)
    edge_img = cv.imread(edge_path, 0)

    edge_img1p = thick_edge2one_pixel(edge_img, edge_th)
    if save_edge_path is not None:
        cv.imwrite(save_edge_path, edge_img1p)

    edge_or_poly = add_edge2poly(edge_img1p, poly_img)
    refine_edge2poly_img, dif_img = refine_edge2poly(edge_or_poly)

    contour_img = np.zeros(poly_img.shape,  np.uint8)
    poly_contours, poly_hierarchy = cv.findContours(poly_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour_img, poly_contours, -1, 255, 1)

    retain_img = contour_img + dif_img
    img_f = retain_img.flatten()
    img_bin_1d = np.where(img_f == 254, 255, 0)
    retain_img_bin = img_bin_1d.reshape(edge_img.shape)

    refine_edge2poly_img = retain_img_bin + refine_edge2poly_img
    img_fill_hole = ndimage.binary_fill_holes(refine_edge2poly_img)

    edge2poly_ret = remove_small_regions(img_as_ubyte(img_fill_hole), area_th)
    cv.imwrite(save_path, edge2poly_ret)


if __name__ == '__main__':
    poly_path = "predicted building region.png"
    edge_path = "predicted building boundary.png"
    save_path = "output of the post-processing method.png"
    save_edge_path = "one-pixel wide edge.png"
    # threshold α in the paper
    edge_th = 200
    # threshold β in the paper
    area_th = 16
    building_ext_post(poly_path, edge_path, save_path, save_edge_path, edge_th, area_th)








