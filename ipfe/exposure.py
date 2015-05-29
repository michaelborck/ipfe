from skimage import img_as_float

def contrast_stretch(image):
    image = img_as_float(image)
    numerator = image - image.min()
    denominator = image.max() - image.min()
    return numerator/denominator
