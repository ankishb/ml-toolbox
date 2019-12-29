# Reference: https://github.com/mdbloice/Augmentor
from PIL import Image, ImageOps, ImageEnhance
import math
from math import floor, ceil
import numpy as np
# from skimage import img_as_ubyte
from skimage import transform

import os
import random
import warnings

def get_all_function():
    all_func = [
        original, RandomContrast, RandomColor, RotateStandard, Flip, Scale,
        Zoom, HistogramEqualisation, Greyscale, Invert, BlackAndWhite,
        RandomBrightness
    ]
    all_func1 = [str(var) for var in all_func]
    print("all augmented function present in thi module are:")
    print(all_func1)
    print("-"*50)

def Original(image, factor=None):
    """ do nothing """
    return image

def RandomContrast(image, factor):
    """Random change the passed image contrast."""
    factor = np.random.uniform(-1*factor, factor)
    
    image_enhancer_contrast = ImageEnhance.Contrast(image)
    return image_enhancer_contrast.enhance(factor)

def RandomColor(image, factor):
    """Random change the passed image saturation."""
    factor = np.random.uniform(-1*factor, factor)

    image_enhancer_color = ImageEnhance.Color(image)
    return image_enhancer_color.enhance(factor)

def RotateStandard(image, factor, resize=True):
    """To perform rotations without automatically cropping the image."""
    random_left = random.randint(factor, 0)
    random_right = random.randint(0, factor)

    left_or_right = random.randint(0, 1)
    rotation = 0

    w, h = image.size
    if left_or_right == 0:
        rotation = random_left
    elif left_or_right == 1:
        rotation = random_right

    image = image.rotate(rotation, expand=expand, resample=Image.BICUBIC)
    if resize:
        image = image.resize((w, h), resample=Image.BICUBIC)

    return image

def RotateRange( image, factor):
    """ This class is used to perform rotations on image by arbitrary numbers of degrees.

    Image are rotated **in place** and an image of the same size is
    returned by this function. That is to say, that after a rotation
    has been performed, the largest possible area of the same aspect ratio
    of the original image is cropped from the skewed image, and this is
    then resized to match the original image size.

    The method by which this is performed is described as follows:

    .. math::
        E = \\frac{\\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}}\\Big(X-\\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}} Y\\Big)}{1-\\frac{(\\sin{\\theta_{a}})^2}{(\\sin{\\theta_{b}})^2}}

    which describes how :math:`E` is derived, and then follows
    :math:`B = Y - E` and :math:`A = \\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}} B`.

    The :ref:`rotating` section describes this in detail and has example
    image to demonstrate this.
    """

    # TODO: Small rotations of 1 or 2 degrees can create black pixels
    factor = int(factor*15)
    random_left = random.randint(factor*(-1), 0)
    random_right = random.randint(0, factor)

    left_or_right = random.randint(0, 1)

    rotation = 0

    if left_or_right == 0:
        rotation = random_left
    elif left_or_right == 1:
        rotation = random_right

    # Get size before we rotate
    x = image.size[0]
    y = image.size[1]

    # Rotate, while expanding the canvas size
    image = image.rotate(rotation, expand=True, resample=Image.BICUBIC)

    # Get size after rotation, which includes the empty space
    X = image.size[0]
    Y = image.size[1]

    # Get our two angles needed for the calculation of the largest area
    angle_a = abs(rotation)
    angle_b = 90 - angle_a

    # Python deals in radians so get our radians
    angle_a_rad = math.radians(angle_a)
    angle_b_rad = math.radians(angle_b)

    # Calculate the sins
    angle_a_sin = math.sin(angle_a_rad)
    angle_b_sin = math.sin(angle_b_rad)

    # Find the maximum area of the rectangle that could be cropped
    E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
        (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
    E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
    B = X - E
    A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

    # Crop this area from the rotated image
    # image = image.crop((E, A, X - E, Y - A))
    image = image.crop((int(round(E)), int(round(A)), int(round(X - E)), int(round(Y - A))))

    # Return the image, re-sized to the size of the image passed originally
    return image.resize((x, y), resample=Image.BICUBIC)

def Resize( image, width, height, resample_filter):
    """This class is used to resize image by absolute values passed as parameters.
    Args:
        resample_filter: [BILINEAR, NEAREST, BICUBIC, ANTIALIAS]
    """
    # TODO: Automatically change this to ANTIALIAS or BICUBIC depending on the size of the file
    return image.resize((width, height), eval("Image.%s" % resample_filter))
                        
def Flip( image, factor=None, top_bottom_left_right="LEFT_RIGHT"):
    """
    Mirror the image according to the `attr`:top_bottom_left_right`
    argument passed to the constructor and return the mirrored image.

    :param top_bottom_left_right: Controls the direction the image should
     be mirrored. Must be one of ``LEFT_RIGHT``, ``TOP_BOTTOM``, or
     ``RANDOM``.

     - ``LEFT_RIGHT`` defines that the image is mirrored along its x axis.
     - ``TOP_BOTTOM`` defines that the image is mirrored along its y axis.
     - ``RANDOM`` defines that the image is mirrored randomly along
       either the x or y axis.

    """

    random_axis = random.randint(0, 1)

    if top_bottom_left_right == "LEFT_RIGHT":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    if top_bottom_left_right == "TOP_BOTTOM":
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    
    if top_bottom_left_right == "RANDOM":
        if random_axis == 0:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        if random_axis == 1:
            return image.transpose(Image.FLIP_TOP_BOTTOM)

    return image

def Crop( image, width, height, centre=True):
    """
    Crop an area from an image, either from a random location or centred,
    using the dimensions supplied during instantiation.
    """

    # All image must be the same size, so we can just check the first \ 
    # image in the list
    w, h = image.size 

    left_shift = random.randint(0, int((w - width)))
    down_shift = random.randint(0, int((h - height)))

    # TODO: Fix. We may want a full crop.
    if width > w or height > h:
        return image

    x1 = left_shift
    y1 = down_shift
    x2 = width + left_shift
    y2 = height + down_shift

    if centre:
        x1 = (w/2) - (width/2)
        y1 = (h/2) - (height/2)
        x2 = (w/2) + (width/2)
        y2 = (h/2) + (height/2)

    return image.crop((x1, y1, x2, y2))

def Scale( image, scale_factor):
    """To increase or decrease image in size by a certain factor, while 
        maintaining the aspect ratio of the original image."""
    w, h = image.size

    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    image = image.resize((w, h), resample=Image.LANCZOS)

    return image

def Zoom(image, factor):
    """to enlarge image (to zoom) but to return a cropped region of the 
    zoomed image of the same size as the original image"""
    factor = round(random.uniform(-1*factor, factor), 2)

    w, h = image.size
    ww = int(round(image.size[0] * factor))
    hh = int(round(image.size[1] * factor))

    image_zoomed = image.resize(
        (ww, hh), resample=Image.BICUBIC
    )
    w_zoomed, h_zoomed = image_zoomed.size
    x1 = floor((float(w_zoomed) / 2) - (float(w) / 2))
    y1 = floor((float(h_zoomed) / 2) - (float(h) / 2))
    x2 = floor((float(w_zoomed) / 2) + (float(w) / 2))
    y2 = floor((float(h_zoomed) / 2) + (float(h) / 2))

    return image_zoomed.crop((x1, y1, x2, y2))

def HistogramEqualisation(image, factor=None):
    """returns the equalised image"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ImageOps.equalize(image)

def Greyscale(image, factor=None):
    """Converts image into having only shades of grey (pixel value 
    intensities) varying from 0 to 255 which represent black and 
    white respectively."""
    return ImageOps.grayscale(image)

def Invert(image, factor=None):
    """Negates the image"""
    return ImageOps.invert(image)

def BlackAndWhite(image, factor):
    """Convert the image passed as an argument to black and white, 1-bit
    monochrome. Uses the threshold passed to the constructor to control 
    the cut-off point where a pixel is converted to black or white.
    """
    # An alternative would be to use
    # PIL.ImageOps.posterize(image=image, bits=1)
    # but this might be faster.
    threshold = int(factor*9)
    image = ImageOps.grayscale(image)
    return image.point(lambda x: 0 if x < threshold else 255, '1')

def RandomBrightness( image, factor):
    """Random change the passed image brightness."""
    factor = np.random.uniform(-1*factor, factor)

    image_enhancer_brightness = ImageEnhance.Brightness(image)
    return image_enhancer_brightness.enhance(factor)

def RandomRotate2(image, k):
    w, h = image.size
    image = transform.rotate(image, angle=k, resize=False, center=None, 
        order=1, mode='constant', cval=0, clip=True, preserve_range=False)

    image = image.resize((w, h), resample=Image.LANCZOS)
    return image





class generate_data():
    """ Augmentation based on percentage and function used are
        [ RandomContrast, RandomColor, RotateRange, Flip, Scale, Zoom, 
          HistogramEqualisation, Greyscale, Invert, BlackAndWhite,
          RandomBrightness ]
    """
    def __init__(self, list_of_files, percentage_aug=0.5):
        self.total = len(list_of_files)
        self.file_path  = list_of_files
        self.percentage = int(percentage_aug * self.total)
        
        self.aug_function = { 
            '1' : RandomContrast, 
            '2' : RandomColor, 
            '3' : RotateRange, 
            '4' : Flip, 
            '5' : Scale, 
            '6' : Zoom, 
            '7' : HistogramEqualisation
            '8' : Greyscale,
            '9' : Invert, 
            '10': BlackAndWhite, 
            '11': RandomBrightness
        }
            
    def augment_data():
        images = []
        for idx, file in enumerate(self.file_path):
            aug_flag = np.random.choice([0,1], size=1, p=[1-self.percentage, self.percentage])
            img = Image.open(file)
            
            if aug_flag == 0:
                image = img
            
            if aug_flag == 1:
                factor = round(np.random.uniform(1, 2),2)
                # select one function randomly out of aug_function
                choice = np.random.choice(len(self.aug_function.items()))
                key_value = list(self.aug_function.keys())[choice]
                function = key_value[1]
                image = function(img, factor)
            
            images.append(image)

        images = np.array(images).astype('float')/255.0
        return images


