"""
This helps in finding the means and standards of the images to normalize before training.

  To run
python3 calculate_means_std.py -i path/to/image/folder/
"""



import argparse
import subprocess
import yaml
import os
import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np


def is_image(filename):
  return any(filename.endswith(ext) for ext in ['.jpg', '.png'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
      '--image', '-i',
      type=str,
      required=True,
      default=None,
      help='Directory to get the images from. If not passed, do from scratch!'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  #
  print("image dir", FLAGS.image)
  print("----------\n")
  
  print("----------\n")

  #

  # create list of images and examine their pixel values
  filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(FLAGS.image)) for f in fn if is_image(f)]

  # examine individually pixel values
  counter = 0.0
  pix_val = np.zeros(3, dtype=np.float)
  for filename in filenames:
    # analize
    print("Accumulating mean", filename)

    # open as rgb
    cv_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # normalize to 1
    cv_img = cv_img.astype(np.float) / 255.0

    # count pixels and add them to counter
    h, w, d = cv_img.shape
    counter += h * w

    # sum to moving pix value counter in each channel
    pix_val += np.sum(cv_img, (0, 1))

  # calculate means
  means = (pix_val / counter)

  # means
  print("means(rgb): ", means)

  # pass again and calculate variance
  pix_var = np.zeros(3, dtype=np.float)
  for filename in filenames:
    # analizel
    print("Accumulating variance", filename)

    # open as rgb
    cv_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # normalize to 1
    cv_img = cv_img.astype(np.float) / 255.0

    # sum to moving pix value counter in each channel
    pix_var += np.sum(np.square(cv_img - means), (0, 1))

  # calculate the standard deviations
  stds = np.sqrt(pix_var / counter)
  print("stds(rgb): ", stds)



  # finalize by printing both
  print("*" * 80)
  print("means(rgb): ", means)
  print("stds(rgb): ", stds)
  print("*" * 80)