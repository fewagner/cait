# import cait as ai
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from tqdm.auto import trange
from ..fit import baseline_template_cubic, pulse_template, scaled_logistic_curve, A_zero
from ..data import get_cc_noise


# ------------------------------------------
# Convenience Functions
# ------------------------------------------

def L2(x):
    """
    TODO

    :param x:
    :type x:
    :return:
    :rtype:
    """
    retval = np.mean(np.abs(x) ** 2, axis=-1)
    return retval


def unfold(dict, idx):
    """
    TODO

    :param dict:
    :type dict:
    :param idx:
    :type idx:
    :return:
    :rtype:
    """
    new_dict = {}
    for k, v in zip(dict.keys(), dict.values()):
        try:
            new_dict[k] = v[idx]
        except:
            new_dict[k] = v
    return new_dict


# ------------------------------------------
# Plotting
# ------------------------------------------

def plot_ev(event):
    """
    TODO

    :param event:
    :type event:
    :return:
    :rtype:
    """
    plt.close()
    plt.subplot(1, 2, 1)
    plt.imshow(icon(event, alpha=1), cmap='Greys')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(recursion(event), cmap='Greys')
    plt.axis('off')
    plt.show()


def plot_events(event_array, t=None, t_unit='s', savepath=None, show=True, text=None):
    """
    TODO

    :param event_array:
    :type event_array:
    :param t:
    :type t:
    :param t_unit:
    :type t_unit:
    TODO
    :return:
    :rtype:
    """
    nmbr_events = event_array.shape[0]
    assert text is None or len(text) == nmbr_events, 'The list of text labels must be as long as the number of events!'
    if t is None:
        t = np.arange(event_array.shape[1])
        xlabel = 'Sample Index'
    else:
        xlabel = 'Time ({})'.format(t_unit)
    assert nmbr_events <= 9, 'Cannot plot more than 9 events!'

    if nmbr_events > 4:
        nmbr_x = 3
        nmbr_y = 3
    elif nmbr_events > 1:
        nmbr_x = 2
        nmbr_y = 2
    else:
        nmbr_x = 1
        nmbr_y = 1

    if nmbr_x == nmbr_y == 1:
        fig, ax = plt.subplots(nmbr_y, nmbr_x, constrained_layout=True)
        ax.plot(t, event_array[0], color='black')
        if text is not None:
            ax.text(x=.6, y=.5, s=text[0], transform=ax.transAxes,
                    bbox=dict(boxstyle='square', fc='white', alpha=0.8, ec='k'))
    else:
        fig, axs = plt.subplots(nmbr_y, nmbr_x, constrained_layout=True)
        for i, ev in enumerate(event_array):
            axs[int(i / nmbr_x), int(i % nmbr_x)].plot(t, ev, color='black')
            if text is not None:
                axs[int(i / nmbr_x), int(i % nmbr_x)].text(x=.6, y=.5, s=text[i],
                                                           transform=axs[int(i / nmbr_x), int(i % nmbr_x)].transAxes,
                                                           bbox=dict(boxstyle='square', fc='white', alpha=0.8, ec='k'))

    fig.supxlabel(xlabel)
    fig.supylabel('Amplitude (V)')

    if savepath is not None:
        plt.savefig(savepath)

    if show:
        plt.show()


def icon(event, size=64, alpha=1):
    """
    TODO

    :param event:
    :type event:
    :param size:
    :type size:
    :param alpha:
    :type alpha:
    :return:
    :rtype:
    """
    img = np.zeros(shape=(size, size), dtype=float)
    xd = np.arange(0, event.shape[0], 1) / event.shape[0] * (size - 1)
    yd = event - np.min(event)
    yd /= np.max(yd)
    yd *= (size - 1)
    for x, y in zip(xd, yd):
        img[size - 1 - int(y), int(x)] += alpha
        if img[size - 1 - int(y), int(x)] > 1:
            img[size - 1 - int(y), int(x)] = 1
    img /= np.max(img)
    return img


def recursion(event, size=64):
    """
    TODO

    :param event:
    :type event:
    :param size:
    :type size:
    :return:
    :rtype:
    """
    img = np.zeros(shape=(size, size))
    event = np.mean(event.reshape((size, int(event.shape[0] / size))), axis=1)
    for i, val in enumerate(event):
        img[:, i] = np.abs(event - val)
    img /= np.max(img)
    return img


# ------------------------------------------
# Shape Templates
# ------------------------------------------

def temp_rise(t, t0, k):
    """
    TODO

    :param t:
    :type t:
    :param t0:
    :type t0:
    :param k:
    :type k:
    :return:
    :rtype:
    """
    ev = np.zeros(t.shape[0])
    ev[t > t0] = k * (t[t > t0] - t0)
    return ev


def temp_spike(t, t0, h, w):
    """
    TODO

    :param t:
    :type t:
    :param t0:
    :type t0:
    :param h:
    :type h:
    :param w:
    :type w:
    :return:
    :rtype:
    """
    ev = np.zeros(t.shape[0])
    ev[np.logical_and(t >= t0, t <= t0 + w)] += h
    return ev


def temp_square(t, h, ivs):
    """
    TODO

    :param t:
    :type t:
    :param h:
    :type h:
    :param ivs:
    :type ivs:
    :return:
    :rtype:
    """
    ev = np.zeros(t.shape[0])
    for iv in ivs:
        ev[np.logical_and(t >= iv[0], t <= iv[1])] += h
    return ev


def temp_jump(t, t0, h, w):
    """
    TODO

    :param t:
    :type t:
    :param t0:
    :type t0:
    :param h:
    :type h:
    :param w:
    :type w:
    :return:
    :rtype:
    """
    ev = np.zeros(t.shape[0])
    if h > 0:
        cond = np.logical_and(t > t0, t < t0 + w)
        ev[cond] = h / w * (t[cond] - t0)
        ev[t > t0 + w] = h
    else:
        ev[t > t0] = h * (1 - np.exp(- (t[t > t0] - t0) / w))
    return ev


# ------------------------------------------
# Default Distributions
# ------------------------------------------

class Distribution:
    """
    TODO
    """

    def sample(self, size, **kwargs):
        """
        TODO

        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplemented


class DefaultPulseHeights(Distribution):
    """
    TODO

    """

    def __init__(self, lamb=0.2, mini=0, maxi=10, p=0.5):
        self.lamb = lamb
        self.mini = mini
        self.maxi = maxi
        self.p = 0.5

    def sample(self, size, **kwargs):
        """
        TODO

        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        pulse_height = np.empty(size, dtype=float)
        rvals = np.random.uniform(size=size)
        pulse_height[rvals < self.p] = np.random.exponential(scale=self.lamb, size=np.sum(rvals < self.p))
        pulse_height[rvals > self.p] = np.random.uniform(low=self.mini, high=self.maxi, size=np.sum(rvals > self.p))

        return pulse_height


class DefaultDrifts(Distribution):
    """
    TODO

    """

    def __init__(self, resolution=None, res_min=0.0005, res_max=0.001):
        self.resolution = resolution
        self.res_min = res_min
        self.res_max = res_max

    def sample(self, size, **kwargs):
        """
        TODO

        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if 'resolution' in kwargs:
            resolution = kwargs['resolution']
        elif self.resolution is None:
            resolution = np.random.uniform(low=self.res_min, high=self.res_max, size=size)
        else:
            resolution = self.resolution

        offsets = np.random.normal(scale=resolution, size=size)
        linear_drifts = np.random.normal(scale=resolution, size=size)
        quadratic_drifts = np.random.normal(scale=resolution, size=size)
        cubic_drifts = np.random.normal(scale=resolution, size=size)

        return offsets, linear_drifts, quadratic_drifts, cubic_drifts


# ------------------------------------------
# Default Class Definitions
# ------------------------------------------

class EventDefinition:
    """
    TODO
    """

    def get_class_pars(self, label, size, **kwargs):
        """
        TODO

        :param label:
        :type label:
        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        raise NotImplemented

        if label == 'Event Pulse':
            pileups = np.ones(size)
            ps_nmbr = 0
            decay = False
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        else:
            raise KeyError('Class {} not available.'.format(label))

        return pileups, ps_nmbr, decay, rise, spike, jump, pulse_reset, tail, onset_iv


class DefaultEventDefinition(EventDefinition):
    """
    TODO

    """

    def get_class_pars(self, label, size, **kwargs):
        """
        TODO

        :param label:
        :type label:
        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if label == 'Event Pulse':
            pileups = np.ones(size)
            ps_nmbr = 0
            decay = False
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Noise':
            pileups = np.zeros(size)
            ps_nmbr = None
            decay = False
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Decaying Baseline':
            pileups = np.zeros(size)  # np.random.choice(a=[0,1], size=1, p=[0.8, 0.2])[0]
            ps_nmbr = None
            decay = True
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Temperature Rise':
            pileups = np.zeros(size)  # np.random.choice(a=[0,1], size=1, p=[0.8, 0.2])[0]
            ps_nmbr = None
            decay = False
            rise = True
            spike = False
            jump = False
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Spike':
            pileups = np.zeros(size)  # np.random.choice(a=[0,1], size=1, p=[0.8, 0.2])[0]
            ps_nmbr = None
            decay = False
            rise = False
            spike = True
            jump = False
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Squid Jump':
            pileups = np.zeros(size)
            ps_nmbr = None
            decay = False
            rise = False
            spike = False
            jump = True
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Reset':
            pileups = np.ones(size)
            ps_nmbr = 0
            decay = False
            rise = False
            spike = False
            jump = False
            pulse_reset = True
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Cosinus Tail':
            pileups = np.ones(size)
            ps_nmbr = 0
            decay = False
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = True
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Decaying Baseline with Event Pulse':
            pileups = np.ones(size)
            ps_nmbr = 0
            decay = True
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Decaying Baseline with Tail Event':
            pileups = np.ones(size)
            ps_nmbr = 0
            decay = True
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = True
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Pile Up':
            pileups = np.random.choice(a=[2, 3, 4, 5], size=size, p=[0.4, 0.3, 0.2, 0.1])
            ps_nmbr = 0
            decay = False
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        elif label == 'Early or late Trigger':
            pileups = np.ones(size)
            ps_nmbr = 0
            decay = False
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = False
            if np.random.uniform(0,1) < 0.5:
                onset_iv = np.array([-160, -40]) * 0.001
            else:
                onset_iv = np.array([40, 400]) * 0.001

        elif label == 'Carrier Event':
            pileups = np.ones(size)
            ps_nmbr = 1
            decay = False
            rise = False
            spike = False
            jump = False
            pulse_reset = False
            tail = False
            onset_iv = np.array([-20, 20]) * 0.001

        else:
            raise KeyError('Class {} not available.'.format(label))

        return pileups, ps_nmbr, decay, rise, spike, jump, pulse_reset, tail, onset_iv


# ------------------------------------------
# Parameter Sampler
# ------------------------------------------

class ParameterSampler:
    """
    TODO

    """

    def __init__(self, record_length, sample_frequency):

        self.record_length = record_length
        self.sample_frequency = sample_frequency
        self.t = (np.arange(record_length) - self.record_length / 4) / self.sample_frequency

        self.args = {
            'ph_dist': DefaultPulseHeights(),
            'resolution': None,
            'pulse_shapes': None,
            'pulse_shapes_probs': None,
            'decay_shapes': None,
            'decay_shapes_probs': None,
            'saturation_pars': None,
            'onset_iv': None,
            'drift_pars': DefaultDrifts(),
            'nps': None,
            'event_definition': DefaultEventDefinition(),
        }

    # ----------------------------------------------------
    # API
    # ----------------------------------------------------

    def set_args(self, **kwargs):
        """
        TODO

        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        for arg, key in zip(kwargs.values(), kwargs.keys()):
            self.args[key] = arg

    def get_event(self,
                  label,
                  size=1,
                  rasterize=True,
                  poly=True,
                  square=False,
                  saturation=True,
                  verb=True):
        """
        TODO

        :param label:
        :type label:
        :param size:
        :type size:
        :param rasterize:
        :type rasterize:
        :param poly:
        :type poly:
        :param square:
        :type square:
        :param saturation:
        :type saturation:
        :param verb:
        :type verb:
        :return:
        :rtype:
        """

        info = {}

        pileups, ps_nmbr, decay, rise, spike, jump, pulse_reset, tail, onset_iv = self.args[
            'event_definition'].get_class_pars(label, size)

        if verb:
            iterator = trange
        else:
            iterator = range

        assert not pulse_reset or all(pileups > 0), 'You must activate at least one pulse to do a reset!'

        event = np.zeros((size, self.record_length))

        if verb:
            print('Sample Noise...')

        # sample baseline
        noise_par, noise_info = self.sample_noise(size=size)
        info['nps'] = noise_par['nps']
        info['resolution'] = noise_info['resolution']

        for i in iterator(size):
            pars = {
                'nps': noise_par['nps'][i] if len(noise_par['nps']) > 1 else noise_par['nps'][0],
                'lamb': noise_par['lamb'][i],
                'nmbr_noise': 1,
                'verb': False
            }
            event[i] = get_cc_noise(**pars)

        # sample polynomial
        if poly:
            if verb:
                print('Sample Polynomials...')
            drift_par = self.sample_drift_par(size=size)
            for i in iterator(size):
                event[i] += baseline_template_cubic(self.t, **unfold(drift_par, i))

        # sample decay
        if decay:
            if verb:
                print('Sample Decay...')
            decay_par = self.sample_decay(size=size)
            for i in iterator(size):
                event[i] += pulse_template(self.t, **unfold(decay_par, i))

        # sample rise
        if rise:
            if verb:
                print('Sample Rise...')
            this_onset = np.random.uniform(onset_iv[0], onset_iv[1], size=size)
            rise_par = self.sample_rise(t0=this_onset, size=size)
            for i in iterator(size):
                event[i] += temp_rise(self.t, **unfold(rise_par, i))

        # sample spike
        if spike:
            if verb:
                print('Sample Spike...')
            this_onset = np.random.uniform(onset_iv[0], onset_iv[1], size=size)
            spike_par = self.sample_spike(size=size, t0=this_onset)
            for i in iterator(size):
                event[i] += temp_spike(self.t, **unfold(spike_par, i))

        # sample square wave
        if square:
            if verb:
                print('Sample Square...')
            square_par = self.sample_square(size=size, resolution=noise_info['resolution'])
            for i in iterator(size):
                event[i] += temp_square(self.t, **unfold(square_par, i))

        # sample jump
        if jump:
            if verb:
                print('Sample Jump...')
            this_onset = np.random.uniform(onset_iv[0], onset_iv[1], size=size)
            jump_par = self.sample_jump(t0=this_onset, size=size)
            for i in iterator(size):
                event[i] += temp_jump(self.t, **unfold(jump_par, i))

        # sample pulses
        highest_pileup = np.max(pileups)
        pulse_model_parameters = ['t0', 'An', 'At', 'tau_n', 'tau_in', 'tau_t']
        if highest_pileup > 0:
            info['pulse_height'] = np.empty(size, dtype=float)
            for key in pulse_model_parameters:
                info[key] = np.empty(size, dtype=float)
            if highest_pileup > 1:
                for i in range(highest_pileup):
                    info['pulse_height_pileup_{}'.format(i)] = np.zeros(size, dtype=float)
                    info['t0_pileup_{}'.format(i)] = np.zeros(size, dtype=float)
        else:
            info['pulse_height'] = np.zeros(size)
            info['t0'] = np.zeros(size)

        for j in range(int(highest_pileup)):
            if verb:
                print('Sample Pulse Nmbr ', j)
            this_onset = np.random.uniform(onset_iv[1],
                                           self.record_length / self.sample_frequency / 4 * 3,
                                           size=size)

            this_onset[j == pileups - 1] = np.random.uniform(onset_iv[0], onset_iv[1],
                                                             size=np.sum(j == pileups - 1))

            pulse_par, pulse_info = self.sample_pulse_par(size=size,
                                                          t0=this_onset,
                                                          ps_nmbr=ps_nmbr)
            for i in range(size):
                if j < pileups[i]:
                    event[i] += pulse_template(self.t, **unfold(pulse_par, i))
                    info['pulse_height_pileup_{}'.format(j)] = pulse_info['pulse_height'][i]
                    info['t0_pileup_{}'.format(j)] = pulse_par['t0'][i]

            cond = j == pileups - 1
            info['pulse_height'][cond] = pulse_info['pulse_height'][cond]
            for key in pulse_model_parameters:
                info[key][cond] = pulse_par[key][cond]

        # pulse reset
        if pulse_reset:
            if verb:
                print('Sample Pulse Reset...')
            pulse_height = np.max(event, axis=1)
            jump_par = self.sample_jump(t0=pulse_par['t0'] + pulse_par['tau_in'],
                                        w=2 * np.minimum(pulse_par['tau_in'], pulse_par['tau_n']),
                                        h=np.random.uniform(-0.9 * pulse_height, -0.3 * pulse_height, size=size))
            for i in iterator(size):
                event[i] += temp_jump(self.t, **unfold(jump_par, i))

        # cosinus tail event
        if tail:
            if verb:
                print('Sample Tail...')
            pulse_height = np.max(event, axis=1)
            h = np.random.uniform(pulse_height * 0.15, 0.4 * pulse_height)
            tail_par = self.sample_jump(t0=pulse_par['t0'],
                                        w=2 * np.maximum(pulse_par['tau_in'], pulse_par['tau_n']),
                                        h=h)
            for i in iterator(size):
                event[i] += temp_jump(self.t, **unfold(tail_par, i))
            info['pulse_height'] += h

        # rasterize
        if rasterize:
            smallest_resolution = 20 / 2 ** 16
            event /= smallest_resolution
            event = np.round(event)
            event *= smallest_resolution

        # saturation
        if saturation:
            saturation_parameters = ['A', 'K', 'C', 'Q', 'B', 'nu']
            if verb:
                print('Sample Saturation ...')
            if self.args['saturation_pars'] is not None:
                sat_par = self.args['saturation_pars']
                event = scaled_logistic_curve(event, **sat_par)
            else:
                sat_par = self.sample_saturation(size=size)
                for i in iterator(size):
                    event[i] = scaled_logistic_curve(event[i], **unfold(sat_par, i))
            for key in saturation_parameters:
                info[key] = sat_par[key]

        return event, info

    # ----------------------------------------------------
    # SAMPLER
    # ----------------------------------------------------

    def sample_pulse_par(self, size=1, **kwargs):
        """
        TODO

        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        pars = {}
        if 't0' in kwargs:
            assert len(kwargs['t0']) == size, 't0 must have length size!'
            pars['t0'] = kwargs['t0']
        else:
            pars['t0'] = np.random.uniform(self.args['onset_iv'][0], self.args['onset_iv'][1], size=size)

        if 'pulse_height' in kwargs:
            pulse_height = kwargs['pulse_height']
        else:
            pulse_height = self.args['ph_dist'].sample(size=size)

        if self.args['pulse_shapes'] is not None:
            if kwargs['ps_nmbr'] is not None:
                shapes = np.repeat(self.args['pulse_shapes'][kwargs['ps_nmbr']],
                                   size).reshape((-1, size))
            elif self.args['pulse_shapes_probs'] is not None:
                assert len(self.args['pulse_shapes']) == len(self.args[
                                                                 'pulse_shapes_probs']), 'Pulse shape probabilites and pulse shapes list must have same length!'
                shapes = np.random.choice(self.args['pulse_shapes'], p=self.args['pulse_shapes_probs'], size=size)
                shapes = np.array(shapes)
            pars['t0'] += shapes[0, :]  # in seconds
            pars['An'] = pulse_height * shapes[1, :]
            pars['At'] = pulse_height * shapes[2, :]
            pars['tau_n'] = shapes[3, :]
            pars['tau_in'] = shapes[4, :]
            pars['tau_t'] = shapes[5, :]

        else:
            shape_ratio = np.random.uniform(0, 1, size=size)
            pars['tau_t'] = np.random.exponential(0.04, size=size)
            pars['tau_in'] = np.random.uniform(0.0005, 0.9 * pars['tau_t'], size=size)
            pars['tau_n'] = np.random.uniform(0.0005, 0.9 * pars['tau_t'], size=size)  # in both components
            pars['An'] = pulse_height * shape_ratio * np.sign(pars['tau_n'] - pars['tau_in'])
            pars['At'] = pulse_height * (1 - shape_ratio)

        for i in range(size):
            arr_max = np.max(pulse_template(t=self.t,
                                            **unfold(pars, i)))

            pars['An'][i] *= (pulse_height[i] / arr_max)
            pars['At'][i] *= (pulse_height[i] / arr_max)

        info = {'pulse_height': pulse_height}
        return pars, info

    def sample_noise(self, size=1, **kwargs):
        """
        TODO

        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        pars = {}
        info = {}
        pars['nmbr_noise'] = size
        fq = np.fft.rfftfreq(n=self.record_length, d=1 / self.sample_frequency)
        info['fq'] = fq
        if self.args['nps'] is not None:
            pars['nps'] = np.copy(self.args['nps'])
            pars['nps'][:, 0] = 0
        else:
            b, a = signal.butter(N=1, Wn=1e4, btype='lowpass', analog=True)
            _, h = signal.freqs(b, a, worN=info['fq'])

            randval = np.random.uniform(0, 0.3)
            alphas = [randval, 2 - 2 * randval]
            coefficients = [1 / 100 ** al for al in reversed(alphas)]
            pars['nps'] = np.sum([c / (info['fq'] + 1) ** a for a, c in zip(alphas, coefficients)], axis=0) * np.abs(h)

            ac_fqs = np.array([50, 150, 250])
            ac_amps = 0.00002 * np.array([1, 0.333, 0.2]) * np.random.uniform(0.2, 1, size=(size, 3))
            ac_offset = np.random.uniform(0, 1, size=size)
            ac_noise = np.zeros((size, self.record_length))
            for i in range(3):
                ac_noise += ac_amps[:, i].reshape((size, 1)) * np.cos(ac_fqs[i].reshape((size, 1)) * 2 * np.pi * (
                        np.tile(self.t, (size, 1)) - ac_offset.reshape((size, 1))))
            pars['nps'] = np.repeat(pars['nps'], size).reshape((size, pars['nps'].shape[0]))
            pars['nps'] += np.abs(np.fft.rfft(ac_noise, axis=1)) ** 2

            pars['nps'][:, 0] = 0

        if self.args['resolution'] is not None:
            resolution = np.copy(self.args['resolution'])
        else:
            resolution = np.random.uniform(0.0005, 0.01, size=size)
        pars['nps'] *= resolution.reshape(-1, 1) ** 2 / np.std(np.sqrt(pars['nps'] / self.record_length), axis=1,
                                                               keepdims=True) ** 2 * 0.65 ** 2

        pars['lamb'] = np.random.uniform(0.01, 0.1, size=size)
        info['resolution'] = resolution
        info['size'] = size
        return pars, info

    def sample_drift_par(self, size=1, **kwargs):
        """
        TODO

        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        pars = {}

        pars['c0'], pars['c1'], pars['c2'], pars['c3'] = \
            self.args['drift_pars'].sample(size,
                                           resolution=self.args['resolution'])
        return pars

    def sample_saturation(self, size=1, **kwargs):
        """
        TODO

        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        pars = {}

        pars['K'] = 10 - np.random.exponential(5, size=size)
        pars['C'] = np.random.gamma(3.3, 0.3, size=size)
        pars['Q'] = np.random.gamma(3.3, 0.3, size=size)
        pars['B'] = np.random.gamma(3.3, 0.3, size=size)
        pars['nu'] = np.random.gamma(3.3, 0.3, size=size)
        pars['A'] = A_zero(K=pars['K'],
                           C=pars['C'],
                           Q=pars['Q'],
                           B=pars['B'],
                           nu=pars['nu'])

        maxi = pars['A'] + (pars['K'] - pars['A']) / pars['C'] ** (1 / pars['nu'])
        cond = np.logical_or(any(maxi < 0.1), any(maxi > 10))
        cond_len = np.sum(cond)

        while cond:
            pars['K'][cond] = 10 - np.random.exponential(5, size=cond_len)
            pars['C'][cond] = np.random.gamma(3.3, 0.3, size=cond_len)
            pars['Q'][cond] = np.random.gamma(3.3, 0.3, size=cond_len)
            pars['B'][cond] = np.random.gamma(3.3, 0.3, size=cond_len)
            pars['nu'][cond] = np.random.gamma(3.3, 0.3, size=cond_len)
            pars['A'][cond] = A_zero(K=pars['K'],
                                     C=pars['C'],
                                     Q=pars['Q'],
                                     B=pars['B'],
                                     nu=pars['nu'])

            maxi = pars['A'] + (pars['K'] - pars['A']) / pars['C'] ** (1 / pars['nu'])
            cond = np.logical_or(any(maxi < 0.1), any(maxi > 10))
            cond_len = np.sum(cond)

        return pars

    def sample_decay(self, min=5, max=30, size=1, t0=None, **kwargs):
        """
        TODO

        :param min:
        :type min:
        :param max:
        :type max:
        :param size:
        :type size:
        :param t0:
        :type t0:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        pars = {}

        # get the minimal value
        if 'minheight' in kwargs:
            minheight = kwargs['minheight'] * np.ones(size)
        elif self.args['resolution'] is not None:
            minheight = 10 * self.args['resolution'] * np.ones(size)
        else:
            minheight = 0.05 * np.ones(size)

        # sample onset
        if t0 is None:
            pars['t0'] = np.random.uniform(-self.record_length / self.sample_frequency,
                                           -self.record_length / self.sample_frequency / 4 * 1.5,
                                           size=size)
        else:
            pars['t0'] = t0

        pulse_height = np.random.uniform(min, max, size=size)

        if self.args['decay_shapes'] is not None and self.args['decay_shapes_probs'] is not None:
            assert len(self.args['decay_shapes']) == len(self.args[
                                                             'decay_shapes_probs']), 'Decay shape probabilites and decay shapes list must have same length!'
            shapes = np.random.choice(self.args['decay_shapes'], p=self.args['decay_shapes_probs'], size=size)
            shapes = np.array(shapes)
            pars['t0'] += shapes[0, :]
            pars['An'] = pulse_height * shapes[1, :]
            pars['At'] = pulse_height * shapes[2, :]
            pars['tau_n'] = shapes[3, :]
            pars['tau_in'] = shapes[4, :]
            pars['tau_t'] = shapes[5, :]

        else:
            shape_ratio = np.random.uniform(0, 1, size=size)
            pars['tau_t'] = np.random.uniform(0.01, 0.2, size=size)
            pars['tau_in'] = np.random.uniform(0.0005, 0.9 * pars['tau_t'], size=size)
            pars['tau_n'] = np.random.uniform(0.0005, 0.9 * pars['tau_t'], size=size)  # in both components
            pars['An'] = pulse_height * shape_ratio * np.sign(pars['tau_n'] - pars['tau_in'])
            pars['At'] = pulse_height * (1 - shape_ratio)

        for i in range(size):

            while True:

                arr_max = pulse_template(t=np.array([-self.record_length / self.sample_frequency / 4]),
                                         **unfold(pars, i))

                if arr_max < minheight[i]:

                    pulse_height = np.random.uniform(min, max, size=1)
                    shape_ratio = np.random.uniform(0, 1, size=1)
                    pars['tau_t'][i] = np.random.uniform(0.01, 0.2, size=1)
                    pars['tau_in'][i] = np.random.uniform(0.0005, 0.9 * pars['tau_t'][i], size=1)
                    pars['tau_n'][i] = np.random.uniform(0.0005, 0.9 * pars['tau_t'][i], size=1)  # in both components
                    pars['An'][i] = pulse_height * shape_ratio * np.sign(pars['tau_n'][i] - pars['tau_in'][i])
                    pars['At'][i] = pulse_height * (1 - shape_ratio)
                else:
                    break

        return pars

    def sample_rise(self, size=1, t0=None):
        """
        TODO

        :param size:
        :type size:
        :param t0:
        :type t0:
        :return:
        :rtype:
        """
        pars = {}

        if t0 is None:
            pars['t0'] = np.random.uniform(-self.record_length / self.sample_frequency / 4,
                                           self.record_length / self.sample_frequency / 3, size=size)
        else:
            pars['t0'] = t0
        pars['k'] = np.random.uniform(0.15, 1.5, size=size)
        return pars

    def sample_spike(self, size=1, t0=None):
        """
        TODO

        :param size:
        :type size:
        :param t0:
        :type t0:
        :return:
        :rtype:
        """
        pars = {}

        if t0 is None:
            pars['t0'] = np.random.uniform(-self.record_length / self.sample_frequency / 4,
                                           self.record_length / self.sample_frequency / 3, size=size)
        else:
            pars['t0'] = t0

        pars['h'] = np.random.uniform(-2, 2, size=size)
        pars['w'] = np.random.uniform(0.0001, 0.001, size=size)
        return pars

    def sample_square(self, size=1, **kwargs):
        """
        TODO

        :param size:
        :type size:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        pars = {}

        if 'resolution' in kwargs:
            pars['h'] = np.random.uniform(kwargs['resolution'], 3 * kwargs['resolution'], size=size)
        elif self.args['resolution'] is not None:
            pars['h'] = np.random.uniform(0.5 * self.args['resolution'], 2 * self.args['resolution'], size=size)
        else:
            pars['h'] = np.random.uniform(0.0005, 0.01, size=size)
        pars['h'] *= [-1, 1][np.random.randint(2)]
        pars['ivs'] = []
        for n in range(size):
            pars['ivs'].append([])
            state = -self.record_length / self.sample_frequency / 4
            while state < self.record_length / self.sample_frequency / 4 * 3:
                for i in range(2):
                    if i == 0:
                        current = [state, ]
                    elif i == 1:
                        current.append(state)
                        pars['ivs'][-1].append(current)
                    state += np.random.uniform(0.01, 0.300)
        return pars

    def sample_jump(self, t0=None, w=None, h=None, size=1):
        """
        TODO

        :param t0:
        :type t0:
        :param w:
        :type w:
        :param h:
        :type h:
        :param size:
        :type size:
        :return:
        :rtype:
        """
        pars = {}

        if t0 is None:
            pars['t0'] = np.random.uniform(-self.record_window / self.sample_frequency / 4,
                                           self.record_window / self.sample_frequency / 3, size=size)
        else:
            pars['t0'] = t0

        if h is None:
            pars['h'] = np.random.choice([-1, 1], size=size) * np.random.uniform(0.1, 10, size=size)
        else:
            pars['h'] = h

        if w is None:
            pars['w'] = np.random.uniform(0.00005, 0.001, size=size)
        else:
            pars['w'] = w

        return pars
