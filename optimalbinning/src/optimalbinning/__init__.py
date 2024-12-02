# -*- coding: utf-8 -*-

'''
Optimal Binning module for Python
======================================
Dr. Ming-Long Lam develops this Python module optimalbinning.

This module performs unsupervised binning on a numeric value.
It finds the optimal bin width based on: 
    Hideaki Shimazaki and Shigeru Shinomoto (2007).
    A Method for Selecting the Bin Size of a Time Histogram,
    Neural Computation, volume 19, issue 6, pages 1503-1527.
    https://www.neuralengine.org/res/histogram.html

Copyright (c) 2024 Ming-Long Lam, Ph.D., Chicago, Illinois, USA. All rights reserved.
'''

import logging

from .binning import (Binning)

logger = logging.getLogger(__name__)

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "0.1.1"

__author__ = "Ming-Long Lam, Ph.D."
__copyright__ = "Copyright (c) Ming-Long Lam, Ph.D., Chicago, Illinois, USA. All rights reserved."
