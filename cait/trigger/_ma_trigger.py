import numpy as np


class MovingAverageTrigger():
    """
    Class to efficiently trigger a time series with moving averages on two time scales.
    TODO this algorithm has potentially huge numerical problems because q must be very close to 1
    """

    def __init__(self, q1, q2, q3, tres, down=1):
        """
        Give parameters for the algorithm

        :param q1: float in (0,1), the weight parameter for past samples in the long time scale
        :param q2: float in (0,1), the weight parameter for past samples in the short time scale
        :param p: float in (0,1), the weight parameter for triggered samples
        :param tres: float, the multiplicative factor for the sigma by which to trigger
        :param down: int, the value how many sample we should average before handing it to the trigger
        """

        if ([q1, q2, p] >= 1).any() or ([q1, q2, p] <= 0).any():
            raise KeyError("Need values in (0,1) for all q!")

        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.tres = tres

        # init the mean, variance, etc
        self.m1 = 0
        self.m2 = 0
        self.sigma = 0

        # init the downsample routine
        self.down = down
        self.x_save = 0
        self.counter = 0
        self.state = 0

    def set_first_value(self, x0):
        """
        Set the first value of the moving average

        :param x0: float, the first value
        :return: -
        """

        self.m1 = x0
        self.m2 = x0

    def get_value(self, x):
        """
        Give a new timestep of the time series to the algo and trigger.
        Calculates moving average m1 on short time scale and moving average m2 and moving std sigma on long time scale,
        when m1 exceed m2 + tresh*sigma or goes below m2 - tresh*sigma, trigger.

        :param x: the next value of the time series
        :return: int, 0 for no trigger, 1 for positive trigger, -1 for negative trigger
        """

        signal = 0

        # calculate the moving average on the short timescale
        self.m2 *= self.q2
        self.m2 += (1 - self.q2) * x

        # set the trigger signal and define
        if self.m2 > self.m1 + self.tres * self.v:
            signal = 1
            q = self.q3
        elif self.m2 < self.m1 - self.tres * self.v:
            signal = -1
            q = self.q3
        else:
            q = self.q2

        # calculate the moving average on the long time scales
        self.m1 *= q
        self.m1 += (1 - q) * x

        # calc the variance from m1 to m2
        self.sigma *= q
        self.sigma += (1 - q) * np.abs(self.m2 - self.m1)

        return signal

    def get_down_value(self, x):
        """
        Wrapper around the get_value routine to downsample first

        :param x: float, the new value
        :return: 0 for trigger, -1 for neg trigger, 1 for pos trigger
        """

        self.x_save += x
        self.counter += 1

        if self.counter >= self.down:
            self.state = self.get_value(self.x_save / self.counter)
            self.x_save = 0
            self.counter = 0

        return self.state
