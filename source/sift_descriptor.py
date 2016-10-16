"""
Python module for use with David Lowe's SIFT code available at:
    http://www.cs.ubc.ca/~lowe/keypoints/
    adapted from the matlab code examples.
Also thanks to http://www.maths.lth.se/matematiklth/personal/solem/downloads/sift.py
author: adarsh
"""

import os
import sys
from skimage.io import load_sift


def bmp_to_key(imagename, outkey):
    """converts <image>.Bmp to <image>.key ASCII file"""

    # check if image file exists
    if not os.path.exists(imagename):
        print "Image file does not exist."
        sys.exit()
    # check if sift binary exists
    if not os.path.exists("./sift"):
        print "sift binary is missing."
        sys.exit()
    # convert bmp to ppm
    command1 = "bmptopnm " + imagename + " > temp.ppm"
    os.system(command1)
    # convert ppm to pgm
    command2 = "ppmtopgm temp.ppm > temp.pgm"
    os.system(command2)
    # convert pgm to key
    command3 = "./sift <temp.pgm >" + outkey
    os.system(command3)
    # clean up
    command4 = "rm -f temp.ppm temp.pgm"
    os.system(command4)
    print "generated", outkey


if __name__ == '__main__':
    bmp_to_key("../trial/1809.Bmp", "1809.key")
    f = open("1809.key")
    my_sift_data = load_sift(f)
    print my_sift_data['data']
