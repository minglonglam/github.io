# -*- coding: utf-8 -*-

'''
Copyright (c) 2024 Ming-Long Lam, Ph.D., Chicago, Illinois, USA. All rights reserved.
'''

import math
import numpy
import pandas

class Binning:
    '''Base class of binning.

    Members
    -------
    bin_width_candidate : (array) a list of bin width values considered.

    list_criterion : (list) a list of the criterion values (ordered according to bin_width_candidate).
    list_n_bin : (list) a list of the number of bins (ordered according to bin_width_candidate).
    list_bin_boundary : (list) a list of the bin boundaries dataframe (ordered according to bin_width_candidate).

    max_n_bin : (int) the maximum number of bins, must be greater than min_n_bin.
    min_n_bin : (int) the minimum number of bins, must be smaller than max_n_bin.

    optimal_criterion : (float) the lowest criterion value.
    optimal_position : (int) the position in the lists corresponding to the lowest criterion value.
    '''

    bin_width_candidate = []

    list_criterion = []
    list_n_bin = []
    list_bin_boundary = []

    max_n_bin = None
    min_n_bin = None

    optimal_criterion = numpy.PINF
    optimal_position = None

    def bin_frequency (self, data = None, bin_lower_boundary = None):

        '''Calculate the frequency (i.e., number of observations) given the bins.
        The bins are left-closed, right-opened intervals

        Parameters
        ----------
        x : (float) the array of non-missing data values
        bin_lower_boundary : (float) the array of bin lower boundaries

        Returns
        -------
        bin_details : (dataframe) An array of bin details, size is (n_bin, 3).  The columns are:
            (0) 'LOWER_CLOSE' : lower-close limits
            (1) 'UPPER_OPEN' : upper-open limits
            (2) 'FREQUENCY' : the number of observations
        '''

        n_bin = len(bin_lower_boundary)

        bin_upper_boundary = numpy.zeros(n_bin)
        bin_upper_boundary[-1] = numpy.PINF

        bin_frequency = numpy.zeros(n_bin)

        if (n_bin >= 2):
            bin_upper_boundary[0:-1] = bin_lower_boundary[1:]

            for u in data:
                qFound = False
                for i in range(n_bin-1):
                    if (u < bin_lower_boundary[i+1]):
                        qFound = True
                        bin_frequency[i] = bin_frequency[i] + 1.0
                        break
                if (not qFound):
                    bin_frequency[n_bin-1] = bin_frequency[n_bin-1] + 1.0

        elif (n_bin == 1):
            bin_frequency[0] = numpy.len(data)

        bin_details = pandas.DataFrame({'LOWER_CLOSE': bin_lower_boundary, \
                                        'UPPER_OPEN': bin_upper_boundary, \
                                        'FREQUENCY': bin_frequency})

        return (bin_details)

    def optimal_binning (self, data = None, min_n_bin = None, max_n_bin = None):
        '''Determine the optimal binning definition.

        Parameters
        ----------
        x : (float) the array of values, cannot contain any missing values
        min_n_bin : (integer) the minimum number of bins. If min_n_bin is not None, it must be smaller than max_n_bin
        max_n_bin : (integer) the maximum number of bins. If max_n_bin is not None, it must be greater than min_n_bin

        Returns
        -------
        n_candidates : (int) number of bin width candidates considered.

        Reference
        ---------
        Hideaki Shimazaki and Shigeru Shinomoto (2007).
            A Method for Selecting the Bin Size of a Time Histogram,
            Neural Computation, volume 19, issue 6, pages 1503-1527.
            https://www.neuralengine.org/res/histogram.html
        '''

        n_candidates = 0

        # Calculate number of values, minimum, maximum, range, and mean of x
        _n_x = 0.0
        _min_x = numpy.PINF
        _max_x = numpy.NINF
        _mean_x = 0.0

        for u in data:
            _n_x = _n_x + 1.0
            _mean_x = _mean_x + u
            if (u > _max_x):
                _max_x = u
            if (u < _min_x):
                _min_x = u

        if (_n_x > 0.0):
            _mean_x = _mean_x / _n_x
            _range_x = _max_x - _min_x
        else:
            raise ValueError('(optimal_binning): The input data array is empty')

        # Specify default values
        if (min_n_bin is None):
            self.min_n_bin = 2
        else:
            self.min_n_bin = min_n_bin

        if (max_n_bin is None):
            self.max_n_bin = _n_x // 2
        else:
            self.max_n_bin = max_n_bin

        if (self.max_n_bin < self.min_n_bin):
            u = self.max_n_bin
            self.max_n_bin = self.min_n_bin
            self.max_n_bin = u

        if (_range_x > 0.0):

            # Determine the minimum and the maximum bin widths
            min_bin_width = math.pow(10.0, math.floor(math.log10(_range_x / self.max_n_bin)))
            max_bin_width = math.pow(10.0, math.ceil(math.log10(_range_x / self.min_n_bin)))

            self.bin_width_candidate = []

            self.list_criterion = []
            self.list_n_bin = []
            self.list_bin_boundary = []

            sequence = 0
            bin_width = min_bin_width
            while (bin_width <= max_bin_width):

                # Generate the boundaries that are nice numbers
                middle_x = bin_width * numpy.round(_mean_x / bin_width)
                n_bin_left = math.ceil((middle_x - _min_x) / bin_width)
                n_bin_right = math.ceil((_max_x - middle_x) / bin_width)
                n_bin = n_bin_left + n_bin_right

                # Ensure the number of bins are within the specified limits
                if (self.min_n_bin <= n_bin <= self.max_n_bin):
                    low_x = middle_x - (n_bin_left - 1) * bin_width

                    bin_lower_boundary = numpy.zeros(n_bin)
                    bin_lower_boundary[0] = numpy.NINF

                    if (n_bin > 2):
                        bin_lower_boundary[1:] = low_x + bin_width * numpy.arange(0, n_bin-1, 1)
                    elif (n_bin == 2):
                        bin_lower_boundary[1] = low_x

                    bin_details = self.bin_frequency(data = data, bin_lower_boundary = bin_lower_boundary)

                    # Compute the Shimazaki and Shinomoto (2007) criterion
                    mean_bin_freq = _n_x / n_bin
                    var_bin_freq = numpy.sum(numpy.power((bin_details['FREQUENCY'].to_numpy() - mean_bin_freq), 2)) / n_bin
                    criterion = (2.0 * mean_bin_freq - var_bin_freq) / bin_width / bin_width

                    n_candidates += 1
                    self.bin_width_candidate.append(bin_width)
                    self.list_criterion.append(criterion)
                    self.list_n_bin.append(n_bin)
                    self.list_bin_boundary.append(bin_details)

                    del bin_details

                # Next bin_width values are multiples of 1, 2, 2.5, and 5
                if (sequence == 1):
                    bin_width = 1.25 * bin_width
                else:
                    bin_width = 2.0 * bin_width

                # Increment sequence by 1 modulus 4
                sequence = sequence + 1
                if (sequence == 4):
                    sequence = 0

            lowest_criterion = numpy.PINF
            lowest_position = None

            i = -1
            for v in self.list_criterion:
                i += 1
                if (v < lowest_criterion):
                    lowest_criterion = v
                    lowest_position = i

            if (lowest_position is not None):
                self.optimal_criterion = lowest_criterion
                self.optimal_position = lowest_position
        else:
            raise ValueError('(optimal_binning): The minimum value equals the maximum value.  Binning cannot be done.')

        return (n_candidates)

    def get_binning_criterion (self):
        '''Return the binning candidates and criteria.

        Parameters
        ----------
        None

        Returns
        -------
        binning_df: (DataFrame) the binning candidates and criteria.
            'BIN_WIDTH': bin width candidates, 'CRITERION': bin criterion,
            'N_BIN': number of bins.
        '''

        output_df = pandas.DataFrame({'BIN_WIDTH': self.bin_width_candidate, \
                                      'CRITERION': self.list_criterion, \
                                      'N_BIN': self.list_n_bin})

        return (output_df.sort_values(by = 'BIN_WIDTH'))

    def get_optimal_boundary (self):
        '''Return the boundaries corresponding to the most optimal binning definition.

        Parameters
        ----------
        None

        Returns
        -------
        optimal_boundary: (DataFrame) the bin boundaries corresponding to the most optimal binning definition.
            'LOWER_CLOSE': bin_lower_boundary, 'UPPER_OPEN': bin_upper_boundary, and 'FREQUENCY': bin_frequency.
        '''

        if (self.optimal_position is not None):
            optimal_boundary = self.list_bin_boundary[self.optimal_position]
        else:
            optimal_boundary = None

        return (optimal_boundary)

    def get_optimal_nbin (self):
        '''Return the number of bins corresponding to the most optimal binning definition.

        Parameters
        ----------
        None

        Returns
        -------
        optimal_nbin: (float) the number of bins corresponding to the most optimal binning definition.
        '''

        if (self.optimal_position is not None):
            optimal_nbin = self.list_n_bin[self.optimal_position]
        else:
            optimal_nbin = 0

        return (optimal_nbin)

    def get_optimal_width (self):
        '''Return the width corresponding to the most optimal binning definition.

        Parameters
        ----------
        None

        Returns
        -------
        optimal_width: (float) the width corresponding to the most optimal binning definition.
        '''

        if (self.optimal_position is not None):
            optimal_width = self.bin_width_candidate[self.optimal_position]
        else:
            optimal_width = None

        return (optimal_width)
