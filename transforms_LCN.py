from __future__ import division
import torch
from torch import Tensor
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from scipy.ndimage.filters import convolve


__all__ = ["LCN_Pinto", "LCN_Jarret"]

class LCN_Pinto(object):
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
        kernel = np.ones((self.kernel_size, self.kernel_size)) / (self.kernel_size**2)

        arr = np.array(tensor)
        local_avg_arr = np.array([convolve(arr[c], kernel, mode=self.mode, cval=self.cval)
                                  for c in range(C)]) # An array that has shape(C, H, W)
                                                      # Each element [c, h, w] is the average of the values
                                                      # in the window that has arr[c,h,w] at the center.

        arr_square = np.square(arr)
        local_avg_arr_square = np.array([convolve(arr_square[c], kernel, mode=self.mode, cval=self.cval)
                                  for c in range(C)]) # An array that has shape(C, H, W)
                                                      # Each element [c, h, w] is the average of the values
                                                      # in the window that has arr_square[c,h,w] at the center.
        local_sum_arr_square = local_avg_arr_square * (self.kernel_size**2)
        local_norm_arr = np.sqrt(local_sum_arr_square) # The tensor of local Euclidean norms.

        local_avg_divided_by_norm = local_avg_arr / local_norm_arr

        result_arr = np.minimum(local_avg_arr, local_avg_divided_by_norm)
        return torch.Tensor(result_arr)



    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, mode={}, cval={})'.format(
            self.kernel_size, self.mode, self.cval)


class LCN_Jarret(object):
    """
    Jarrett, Kevin & Kavukcuoglu, Koray & Ranzato, Marc'Aurelio & Lecun, Yann. (2009).
     What is the Best Multi-Stage Architecture for Object Recognition?.
     In Proc Intl Conf on Comput Vis. 12. 10.1109/ICCV.2009.5459469.

    the kernel size is controllable by argument kernel_size.
    """

    def __init__(self, kernel_size=3, sigma=1, mode='constant', cval=0.0):
        """

        :param kernel_size: int, kernel(window) size for local region of image.
        :param sigma: float, the standard deviation of the Gaussian kernel for weight window.
        :param mode: {'reflect', 'constant', 'nearest', 'mirror', 'warp'}, optional
                        determines how the array borders are handled. The meanings are listed in
                        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.convolve.html
                        default is 'constant' as 0, different from the scipy default.
        """

        def gaussian_kernel(size, sigma):
            """
            :return: size x size shape Gaussian kernel with standard deviation sigma
            """
            side = int((size - 1) // 2)
            x, y = np.mgrid[-side:side + 1, -side:side + 1]
            g = np.exp(-(x ** 2 / (sigma ** 2 * float(side)) + y ** 2 / (sigma ** 2 * float(side))))
            return g / np.sum(g)

        self.mode = mode
        self.cval = cval
        self.kernel = gaussian_kernel(kernel_size, sigma)

    def __call__(self, tensor):
        """

        :param tensor: Tensor image os size (C, H, W) to be normalized.
        :return:
            Tensor: Normalized Tensor image, in size (C, H, W).
        """

        C, H, W = tensor.size()

        arr = np.array(tensor)
        arr_v = arr - np.sum(np.array([convolve(arr[c], self.kernel, mode=self.mode, cval=self.cval)
                                       for c in range(C)]), axis=0)
        # arr_v = arr - np.array([convolve(arr[c], self.kernel, mode=self.mode, cval=self.cval)
        #                           for c in range(C)])

        arr_sigma_square_stack = np.array([convolve(np.square(arr_v[c]), self.kernel, mode=self.mode, cval=self.cval)
                                           for c in range(C)])  # An array that has shape (C, H, W).
        # (c, h, w) element is the result of
        # convolution, at (h, w) coordinates,
        # with the Gaussian kernel(2 dim)
        # and arr_v[c] as input

        arr_sigma = np.sqrt(np.sum(arr_sigma_square_stack, axis=0))
        c = np.mean(arr_sigma)
        arr_v_divided_by_c = arr_v / c
        arr_v_divided_by_arr_sigma = arr_v / (arr_sigma)
        arr_y = np.maximum(arr_v_divided_by_c, arr_v_divided_by_arr_sigma)
        return torch.Tensor(arr_y)

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, sigma={1}, mode={2}, cval={3})'.format(
            self.kernel_size, self.sigma, self.mode, self.cval)

