import torch
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import cv2

def get_VOC_colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap
N = 256
voc_cmap = get_VOC_colormap(N=N, normalized=True)
voc_cmap = LinearSegmentedColormap.from_list('voc_cmap', voc_cmap, N=N)

def tensormask_to_nprgba(mask, alpha=0.5):
    rgba = voc_cmap(mask.cpu().numpy())
    rgba[:, :, -1] = alpha
    return voc_cmap(mask.cpu().numpy())

def stack_image_and_pred_mask(I, M):
    fig, axarr = plt.subplots( 1, 2 )
    axarr[0].imshow(I.cpu().permute(1, 2, 0) )
    axarr[1].imshow( tensormask_to_nprgba( M ) )
    # plt.show()
    return fig

def overlay_mask_on_image(I,M):
    fig = plt.figure()
    ax = plt.gca()
    Is = I.cpu().permute( 1, 2, 0 ).numpy()
    Is = cv2.cvtColor(Is, cv2.COLOR_RGB2GRAY)
    z = np.ones((Is.shape[0], Is.shape[1], 4))
    z[:, :, 0:3] = np.repeat(np.expand_dims(Is, 2), 3, 2)
    mask_rgba = tensormask_to_nprgba( M )
    Iout = z[:, :, 0:3] * z[:, :, 3:4] + mask_rgba[:, :, 0:3] * mask_rgba[:, :, 3:4]
    ax.imshow(Iout)
    # plt.show()
    return fig, ax