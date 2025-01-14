# Copyright (C) 2025 Matthias Deiml, Daniel Peterseim - All rights reserved

from PIL import Image
import numpy as np
from matplotlib.cm import get_cmap
import cmocean

image_folder = "./images/"


def save_plot(
    data,
    filename,
    max=None,
    cm=get_cmap("cmo.balance"),
    negative_gray=False,
    upsample=False,
):
    """
    Convenience function for exporting image plots
    """
    if max is None:
        max = np.max(np.abs(data))
    im_data = cm(data / max / 2 + 0.5, bytes=True)
    im = Image.fromarray(im_data)
    im.save(image_folder + filename)
