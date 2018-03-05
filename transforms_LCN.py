from __future__ import division
import torch
from torch import Tensor
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from scipy.ndimage.filters import convolve

from . import functional as F

__all__ = ["LocalContrastNormalization"]

class LocalContrastNormalization(object):
    """
    Conduct a local contrast normalization algorithm by
    Pinto, N., Cox, D. D., and DiCarlo, J. J. (2008). Why is real-world visual object recognition hard?
     PLoS Comput Biol , 4 . 456 (they called this "Local input divisive normalization")

    the kernel size is controllable by argument kernel_size.
    """
    def __init__(self, kernel_size=3, mode='constant', cval=0.0):
        """

        :param kernel_size: int, kernel(window) size for local region of image.
        :param mode: {'reflect', 'constant', 'nearest', 'mirror', 'warp'}, optional
                        determines how the array borders are handled. The meanings are listed in
                        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.convolve.html
                        default is 'constant' as 0, different from the scipy default.

        """
        self.kernel_size = kernel_size
        self.mode = mode
        self.cval = cval

    def __call__(self, tensor):
        """

        :param tensor: Tensor image os size (C, H, W) to be normalized.
        :return:
            Tensor: Normalized Tensor image, in size (C, H, W).
        """
        C, H, W = tensor.size()
        kernel = np.ones((self.kernel_size, self.kernel_size))

        arr = np.array(tensor)
        local_sum_arr = np.array([convolve(arr[c], kernel, mode=self.mode, cval=self.cval)
                                  for c in range(C)]) # An array that has shape(C, H, W)
                                                      # Each element [c, h, w] is the summation of the values
                                                      # in the window that has arr[c,h,w] at the center.
        local_avg_arr = local_sum_arr / (self.kernel_size**2) # The tensor of local averages.

        arr_square = np.square(arr)
        local_sum_arr_square = np.array([convolve(arr_square[c], kernel, mode=self.mode, cval=self.cval)
                                  for c in range(C)]) # An array that has shape(C, H, W)
                                                      # Each element [c, h, w] is the summation of the values
                                                      # in the window that has arr_square[c,h,w] at the center.
        local_norm_arr = np.sqrt(local_sum_arr_square) # The tensor of local Euclidean norms.


        local_avg_divided_by_norm = local_avg_arr / local_norm_arr

        result_arr = np.minimum(local_avg_rr, local_avg_divided_by_norm)
        return torch.Tensor(result_arr)



    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, threshold={1})'.format(self.kernel_size, self.threshold)