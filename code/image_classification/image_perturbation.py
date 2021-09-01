# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: image_perturbation
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
from skimage import transform
import random

def image_perturbation(image):
    random_angle = random.randint(0,360)
    perturbed_image = transform.rotate(image, angle=random_angle)
    return perturbed_image