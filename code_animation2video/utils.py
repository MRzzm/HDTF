import numpy as np
import cv2

def make_coordinate_grid(image_size):
    h, w = image_size
    x = np.arange(w)
    y = np.arange(h)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    xx = x.reshape(1, -1).repeat(h, axis=0)
    yy = y.reshape(-1, 1).repeat(w, axis=1)
    meshed = np.stack([xx, yy], 2)
    return meshed

def visualize_dense_flow(dense_flow):
    hsv = np.zeros([dense_flow.shape[0], dense_flow.shape[1]])
    hsv = np.stack([hsv, hsv, hsv], 2).astype(np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(dense_flow[:, :, 0], dense_flow[:, :, 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = rgb.astype(np.uint8)
    return rgb