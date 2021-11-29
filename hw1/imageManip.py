import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None
    
    # YOUR CODE HERE
    out = io.imread(fname = image_path)
    # END YOUR CODE
   
    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    
    # YOUR CODE HERE
    out = 0.5 * image.astype(np.float64) ** 2
    # END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None
    
    # YOUR CODE HERE
    out = color.rgb2gray(image)
    # EBD YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    
    out = None
    
    # YOUR CODE HERE
    out = image.copy()
    order = ["R", "G", "B"]
    
    channel = order.index(channel)
    out[:, :, channel] = 0
    # END YOUR CODE
    
    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    out = None

    ### YOUR CODE HERE
    channel_order = ['H', 'S', 'V']
    channel = channel_order.index(channel)
    out = color.rgb2hsv(image)
    
    out = out[:, :, channel]
    ### END YOUR CODE
    
    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    
    ### YOUR CODE HERE  
    image1 = rgb_exclusion(image1, channel1)
    image2 = rgb_exclusion(image2, channel2)
    
    #Get left half of image1
    image1_half_cols = image1.shape[1]//2
    image1_half = image1[:,:image1_half_cols,:]
    
    #Get right half of image2
    image2_half_cols = image2.shape[1]//2
    image2_half = image2[:,image2_half_cols:,:]
    
    #Get final image
    out = np.concatenate((image1_half, image2_half), axis=1)
    ### END YOUR CODE
    
    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    image_half_rows = image.shape[1]//2
    image_half_cols = image.shape[1]//2
    
    # Separate four quadrants
    top_left = image[:image_half_rows,:image_half_cols,:]
    top_right = image[:image_half_rows, image_half_cols:, :]
    bottom_left = image[image_half_rows:,:image_half_cols, :]
    bottom_right = image[image_half_rows:, image_half_cols:, :]
    
    # Modify the quadrants
    top_left_mod = rgb_exclusion(top_left, 'R')
    top_right_mod = dim_image(top_right)
    bottom_left_mod = bottom_left ** 0.5
    bottom_right_mod = rgb_exclusion(bottom_right, 'R')
    
    # Join the image
    top = np.concatenate((top_left_mod, top_right_mod), axis=1)
    bottom = np.concatenate((bottom_left_mod, bottom_right_mod), axis=1)
    
    out = np.concatenate((top, bottom), axis=0)
    ### END YOUR CODE

    return out
