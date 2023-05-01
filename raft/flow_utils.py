# Python program to illustrate
# foreground extraction using
# GrabCut algorithm

# organize imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


def generate_warped_image(flo, image1, image2, idx):
    # import cv2
    # flo = flo[0].permute(1,2,0).cpu().numpy()
    # image1 = image1[0].permute(1,2,0).cpu().numpy()
    # image2 = image2[0].permute(1,2,0).cpu().numpy()

    h, w = flo.shape[:2]
    flo[:,:,0] += np.arange(w)
    flo[:,:,1] += np.arange(h)[:,np.newaxis]
    warped_img = cv2.remap(image1, flo, None, cv2.INTER_LINEAR)
    residual = np.abs(warped_img - image2)
    residual = np.mean(residual, axis=2)

    return warped_img, residual

def rgbtoint32(rgb):
    color = 0
    for c in rgb[::-1]:
        color = (color<<8) + c
        # Do not forget parenthesis.
        # color<< 8 + c is equivalent of color << (8+c)
    return color

def int32torgb(color):
    rgb = []
    for i in range(3):
        rgb.append(color&0xff)
        color = color >> 8
    return rgb

def get_color_wheel_distance(flo, idx):

    def rgb_to_hsv(rgb):
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

        # h, s, v = hue, saturation, value
        cmax = max(r, g, b) # maximum of r, g, b
        cmin = min(r, g, b) # minimum of r, g, b
        diff = cmax-cmin	 # diff of cmax and cmin.

        # if cmax and cmax are equal then h = 0
        if cmax == cmin:
            h = 0

        # if cmax equal r then compute h
        elif cmax == r:
            h = (60 * ((g - b) / diff) + 360) % 360

        # if cmax equal g then compute h
        elif cmax == g:
            h = (60 * ((b - r) / diff) + 120) % 360

        # if cmax equal b then compute h
        elif cmax == b:
            h = (60 * ((r - g) / diff) + 240) % 360

        # if cmax equal zero
        if cmax == 0:
            s = 0
        else:
            s = (diff / cmax) * 100

        # compute v
        v = cmax * 100
        # normalize h,s,v
        h = h/360
        s = s/100
        v = v/100
        return h, s, v

    def compute_distance(hsv):
        h1, s1, v1 = hsv
        h2, s2, v2 = rgb_to_hsv((255, 255, 255)) # white

        distance = math.pow((math.sin(h1)*s1*v1 - math.sin(h2)*s2*v2),2) \
            + math.pow((math.cos(h1)*s1*v1 - math.cos(h2)*s2*v2),2) \
            + math.pow((v1 - v2),2)
        return distance

    # get the distance of a color from origin in colorwheel
    mask = np.zeros(flo.shape[:2], np.uint8)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    if idx < 155:
        startingPoint_x, startingPoint_y = 380, 120
        rectangle = (startingPoint_x, startingPoint_y+idx+idx//2, 250, 250)
    else:
        # print('here')
        #TODO: Find the change in starting point
        startingPoint_x, startingPoint_y = 300, 600
        rectangle = (startingPoint_x+idx+idx//2, startingPoint_y, 250, 250)

    # bbox = cv2.rectangle(
    #     flo, (startingPoint_x+idx, startingPoint_y), \
    #     (startingPoint_x+idx+250, startingPoint_y+250), \
    #     (0, 255, 0), 2
    # )
    # cv2.imwrite('bbox.jpg', bbox)
    # return

    cv2.grabCut(flo, mask, rectangle,
			backgroundModel, foregroundModel,
			3, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    flo_mask = flo * mask[:, :, np.newaxis]
    flo_mask = flo_mask[startingPoint_y+idx:startingPoint_y+idx+250,\
               startingPoint_x+idx:startingPoint_x+idx+250, :]
    flo_mask = cv2.cvtColor(flo_mask, cv2.COLOR_BGR2RGB)
    indices = (flo_mask > 0).nonzero()
    color = np.array(
        [[rgbtoint32(flo_mask[indices[0][i], indices[1][i], :])] \
         for i in range(len(indices[0]))]
    )
    unique, counts = np.unique(color, return_counts=True)
    freq = np.asarray((unique, counts)).T
    max_freq = freq[np.argsort(freq[:, 1])[:5]]
    x, y, h = 800, 100, 40

    for i in range(len(max_freq)):
        rgb = int32torgb(max_freq[i][0])
        rgb = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        distance = round(compute_distance(rgb_to_hsv(rgb)), 3)
        flo = cv2.rectangle(
            flo, (x, y), (x + h, y + h), color=rgb, thickness=-1
        )
        cv2.putText(flo, str(distance), (x+80, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        y += h + 10

    return flo
    # cv2.imwrite(f'flo/{idx}.jpg', flo)
