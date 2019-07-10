import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pdf_functions as pf
import data_generator as dg
import distribution_configuration
from src import random_behavior


def _evenly_spaced_points(lower_bound, upper_bound, number_of_animation_points):
    width = upper_bound - lower_bound
    # Generate equally spaced animation points
    evenly_spaced_points = np.arange(0, number_of_animation_points) / float(number_of_animation_points)
    evenly_spaced_points = lower_bound + evenly_spaced_points * float(width)
    return evenly_spaced_points.astype(np.float32)


conf = distribution_configuration.get_configuration()
pdf_functions = pf.PdfFunctions(conf)
random = random_behavior.Random()
data_generator = dg.DataGenerator(conf, pdf_functions, random)
a_star, _ = data_generator.sample_gaussian_mixture(conf.r)

number_of_animation_points = 1000
upper_bound = conf.d * 4
lower_bound = 0
evenly_spaced_distances = _evenly_spaced_points(lower_bound, upper_bound, number_of_animation_points)
# Draw random points and scale to fit distances
animation_points, _ = data_generator.sample_gaussian_mixture(number_of_animation_points)
sampled_distance = np.sum(animation_points ** 2.0, axis=1) ** 0.5
animation_points = animation_points / sampled_distance.reshape(number_of_animation_points, 1)
animation_points = animation_points * evenly_spaced_distances.reshape(number_of_animation_points, 1) ** 0.5

scaler = 2
H_inverse = np.eye(conf.d).astype(np.float32) / scaler
fa_tensor = pdf_functions.chi_square_kde_centered_exponent(H_inverse, animation_points, a_star, number_of_animation_points, 0.001)

pa_tensor = pdf_functions.chi_squared_distribution(evenly_spaced_distances)
session = tf.Session()
fa = session.run(fa_tensor) #/ 3.148
pa = session.run(pa_tensor)
width = (upper_bound - lower_bound) / float(number_of_animation_points)
#print fa[0]
print np.sum(fa * width)
print np.sum(pa * width)
# plt.plot(np.sum(animation_points ** 2.0,axis=1), fa)
plt.plot(np.sum(animation_points ** 2.0,axis=1), pa)
gauss = np.exp(-evenly_spaced_distances / conf.d) / 50
plt.plot(evenly_spaced_distances, gauss)
print np.sum(gauss * width)
plt.show()
