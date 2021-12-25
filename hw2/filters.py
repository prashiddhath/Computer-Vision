import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #Padding image with zeros to operate on edge pixels
    padded_img = np.zeros((Hi+Hk-1, Wi+Wk-1))
    
    #Copying image inside the padding
    h = Hk//2
    w = Wk//2
    padded_img[h:-h, w:-w]=image
    
    #Flipping the kernel
    kernel = kernel[::-1]
    kernel = np.fliplr(kernel)
    
    for m in range(Hi):
        for n in range(Wi):
            for k in range(Hk):
                for l in range(Wk):
                    out[m,n] = out[m,n] + (padded_img[m+k][n+l] * kernel[k][l])
    
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    new_H = H + 2*pad_height
    new_W = W + 2*pad_width
    
    out = np.zeros((new_H, new_W))
    
    out[pad_height:pad_height+H, pad_width:pad_width+W] = image

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    #Padding
    h_pad = Hk//2
    w_pad = Wk//2
    padded_img = zero_pad(image, h_pad, w_pad)
    
    #Flipping the kernel
    kernel = kernel[::-1]
    kernel = np.fliplr(kernel)
    
    for m in range(Hi):
        for n in range(Wi):
            section = padded_img[m:m+Hk, n:n+Wk]
            out[m, n] = (section * kernel).sum()

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image = zero_pad(image, Hk//2, Wk//2)
    
    kernel = kernel[::-1]
    kernel = np.fliplr(kernel)
    
    temp = np.zeros((Hi*Wi, Hk*Wk))
    
    for i in range(Hi*Wi):
        m = i // Wi
        n = i % Wi
        temp[i, :] = image[m: m+Hk, n: n+Wk].reshape(1, Hk*Wk)
        
    out = temp.dot(kernel.reshape(Hk*Wk, 1)).reshape(Hi, Wi)

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    
    #Flipping the kernel
    g = np.fliplr(np.flipud(g))
    out = conv_fast(f, g)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    
    g_mean = np.mean(g)
    g_zero = g - g_mean
    
    '''
        The question didn't state to use the mean of f but I assume that zero cross correlation actually 
        subtracts the zero mean from both f and g. Just adding it as comments, in case it is actually required.
    '''
    # f_mean = np.mean(f)
    # f_zero = f - f_mean
    # out_zero = cross_correlation(f_zero, g_zero)
    
    out = cross_correlation(f, g_zero)

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    g_norm = (g - np.mean(g)) / np.std(g)
    
    #Padding
    f = zero_pad(f, Hg//2, Wg//2)
    
    for m in range(Hf):
        for n in range(Wf):
            section = f[m:m+Hg, n:n+Wg]
            section_norm = (section - np.mean(section)) / np.std(section)
            out[m, n] = (section_norm * g_norm).sum()

    return out