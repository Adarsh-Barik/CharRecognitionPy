"""
Python module for use with David Lowe's SIFT code available at:
    http://www.cs.ubc.ca/~lowe/keypoints/
    adapted from the matlab code examples.
Also thanks to http://www.maths.lth.se/matematiklth/personal/solem/downloads/sift.py
author: adarsh
"""
import os
import subprocess
from os import path, system
import sys
from skimage.io import load_sift


def bmp_to_key(imagename, outkey):
    """converts <image>.Bmp to <image>.key ASCII file"""

    # check if image file exists
    if not path.exists(imagename):
        print ("Image file does not exist.")
        sys.exit()
    # check if sift binary exists
    if not path.exists("./sift"):
        print ("sift binary is missing.")
        sys.exit()
    # convert bmp to ppm
    command1 = "bmptopnm " + imagename + " > temp.ppm"
    system(command1)
    # convert ppm to pgm
    command2 = "ppmtopgm temp.ppm > temp.pgm"
    system(command2)
    # convert pgm to key
    if os.name == "posix":
        command3 = "./sift <temp.pgm >" + outkey
    else:
        command3 = "siftWin32 <temp.pgm >" + outkey
    system(command3)
    # clean up
    command4 = "rm -f temp.ppm temp.pgm"
    system(command4)
    print ("generated", outkey)

def key_to_descriptor_array(keyfile):
    """ changes keys to an array """
    if not path.exists(keyfile):
        print ("Key file doesn't exist.")
        sys.exit()
    f = open(keyfile)
    my_sift_data = load_sift(f)
    f.close()
    return my_sift_data['data']


if __name__ == '__main__':
    bmp_to_key("../trial/1809.Bmp", "1809.key")
    print (key_to_descriptor_array("1809.key"))
