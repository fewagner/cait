import json
import os
import re
import warnings
from typing import List, Tuple, Union

import numpy as np
import scipy as sp

import cait as ai
import cait.versatile as vai

from ..iterators.iteratorbase import IteratorBaseClass
from .arraywithbenefits import ArrayWithBenefits
from .nps import NPS
from .of import OF
from .sev import SEV

_supported_bl_models = ['poly', 'exp', 'exp_dtau', 'custom']

def inner_product(x_tilde, y_tilde, nps, dt):
    """The noise-weighted inner product between x and y in frequency domain."""
    w = np.ones_like(x_tilde)
    w[1:-1] = 2
    N = 2*(len(w) - 1)
    return 2*dt/N * np.vdot(x_tilde, w*y_tilde/nps).real

def construct_basis(
        sev: SEV,
        basis: str, 
        bspec: Union[int, Tuple[List[float], bool], List[List[float]]],
        ):
    """Construct the basis for the generalized optimum filter after validating the inputs (which are the same as for the constructor of GOF)."""
    if not basis.lower() in _supported_bl_models:
        raise ValueError(f"Input argument has to be one of {_supported_bl_models}. Got {basis}.")
        
    # Input validation and definition of correct basis
    if basis.lower() == "poly":
        if not isinstance(bspec, (int, np.integer)):
            raise TypeError(f"For basis model 'poly', the argument 'bspec' must be a non-negative integer (the polynomial degree). Got {type(bspec).__name__}.")
        if bspec < 0:
            raise ValueError(f"For basis model 'poly', the argument 'bspec' must be a non-negative integer (the polynomial degree). Got {bspec}.")
        
        return [np.linspace(-1, 1, sev.shape[-1])**k for k in range(bspec + 1)]

    elif basis.lower() in ["exp", "exp_dtau"]:
        if not isinstance(bspec, (list, tuple, np.ndarray)):
            raise TypeError(f"For basis model '{basis.lower()}', the argument 'bspec' must be a tuple/list/np.ndarray containing floats. Got {type(bspec).__name__}.")
        if not all([isinstance(x, (float, np.floating)) for x in bspec]):
            raise TypeError(f"For basis model '{basis.lower()}', the argument 'bspec' must be a tuple/list/np.ndarray containing floats. Got {bspec}.")
        
        t = np.arange(sev.shape[-1])*sev.dt_us*1e-6 # s
        B = [np.ones(sev.shape[-1])] + [np.exp(-t/tau) for tau in bspec]

        if basis.lower() == "exp_dtau":
            B += [t*np.exp(-t/tau) for tau in bspec]

        return B

    elif basis.lower() == "custom":
        if not isinstance(bspec, (list, tuple, np.ndarray)):
            raise TypeError(f"For basis model 'custom', the argument 'bspec' must be a tuple/list/np.ndarray containing arrays that specify the basis. Got {type(bspec).__name__}.")
        _basis_lens = [len(x) for x in bspec]
        if len(set(_basis_lens)) > 1:
            raise ValueError(f"For basis model 'custom', the argument 'bspec' must be a tuple/list/np.ndarray containing equally sized arrays. Got lengths {_basis_lens}.")
        
        return [np.array(x) for x in bspec]

def gram_matrix(B_tilde: List[List[float]], nps: np.ndarray, dt: float):
    """Construct the Gram matrix for a basis 'B' using the frequency domain inner product defined by 'nps' and 'dt' (in seconds!)."""
    K = len(B_tilde)
    # Calculate Gram matrix
    G = np.zeros((K, K))
    for j in range(K):
        for l in range(K):
            G[j,l] = inner_product(B_tilde[j], B_tilde[l], nps, dt)

    return G

class BCC(ArrayWithBenefits):
    """Object representing Basis Coefficient Covariance matrix"""
    def __init__(
            self, 
            it: IteratorBaseClass,
            nps: NPS,
            *,
            basis: str = "poly",
            bspec: Union[int, Tuple[List[float], bool], List[List[float]]] = 3,
            ):
        if not isinstance(it, IteratorBaseClass):
            raise TypeError(f"Input argument 'it' must be an event iterator, not {type(it).__name__}.")
        if not isinstance(nps, NPS):
            raise TypeError(f"Input argument 'nps' has to be of type 'NPS', not {type(nps).__name__}.")
        
        if it.dt_us != nps.dt_us:
            raise ValueError(f"Input arguments 'it' and 'nps' must have the same timebase (dt_us). Got {it.dt_us} and {nps.dt_us}.")
        
        if 2*(nps.shape[0]-1) != it.record_length:
            raise ValueError(f"For iterators using record_length N, the NPS must have shape (N//2-1,). Got N={it.record_length} and {nps.shape}.")
        
        B = construct_basis(
            # SEV is only needed to infer dt_us and the length of the arrays.
            sev=SEV(np.ones(2*(nps.shape[0]-1)), dt_us=nps.dt_us), 
            basis=basis, 
            bspec=bspec,
        )
        K = len(B)
        dt = nps.dt_us*1e-6
        B_tilde = [np.fft.rfft(b) for b in B]
        G = gram_matrix(B_tilde=B_tilde, nps=nps, dt=dt)
        G_inv = sp.linalg.solve(G, np.eye(K), assume_a="her")
        
        P_tilde = G_inv @ np.array([2*dt*b/nps for b in B_tilde])
        P = np.fft.irfft(P_tilde, axis=-1)

        # Calculate ML estimators for all baseline coefficients
        beta_hat = vai.apply(lambda x: P.dot(x), it.with_processing(lambda x: x - x[0]))
        beta_hat_centered = beta_hat - np.mean(beta_hat, axis=0, keepdims=True)

        # Cov(beta_hat) = Cov(beta) + G_inv
        # https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
        cov_beta_hat = beta_hat_centered.T.dot(beta_hat_centered)/(len(it) - 1)
        cov_beta = cov_beta_hat - G_inv
        
        self._bcc = cov_beta
        self._dt_us = nps.dt_us
        
    @property
    def _array(self):
        return self._bcc

    @_array.setter
    def _array(self, array):
        if self._bcc.size == 0:
            self._bcc = array
        else:
            raise Exception("BCC._array can only be set as long as it is empty.")

    @property
    def dt_us(self):
        return self._dt_us

class GOF(OF):
    """
    Object representing a generalized optimum filter (see P. Schreiner et al. 2026, to be published).

    The generalized filter has to be constructed form a template (SEV) and a noise power spectrum (NPS), as well as a basis which describes the baseline. If you intend to build a traditional optimum filter, use :class:`cait.versatile.OF` instead. 

    Note that ``vai.OF(sev, nps)`` is equivalent to ``vai.GOF(sev, nps, basis='poly', bspec=0, phase='argmax')``.

    :param sev: The template to be used for the filter. Note that only single channel SEVs are supported, i.e. has to be of shape ``(N,)``.
    :type sev: SEV
    :param nps: The noise power spectrum to be used for the filter. Note that only single channel NPSs are supported. Has to have shape ``(N//2+1,)``.
    :type nps: NPS
    :param basis: The basis for the baseline description to be used. Choose either of **poly** (polynomial baseline), **exp** (exponential baseline, plus a constant), or **custom** (arbitrary basis). The specifics of the corresponding basis are handled by the ``bspec`` argument. Defaults to 'poly'.
    :type basis: str, optional
    :param bspec: Detailed specifications of ``basis``. See table below. Defaults to 3. Together with the default of ``basis``, this corresponds to cubic polynomials.
    :type bspec: Union[int, Tuple[List[float], bool], List[List[float]]], optional
    :param bcc: Basis Coefficient Covariance matrix. If given, the ... Has to have shape (K, K), where K is the number of baseline basis vectors. Defaults to None, i.e. implicitly assuming no knowledge about coefficients = infinite variance.
    :type bcc: np.ndarray, optional
    :param window_sev: If True, the standard :class:`cait.versatile.TukeyWindow` is applied to the SEV before filter creation. This might especially be relevant if the SEV does not fully decay inside the record window as it reduces high frequency pollution from the resulting step. Note that applying a window will always increase the minimum possible resolution, but the tradeoff of having less numerical artifacts might be worth it. Defaults to False.
    :type window_sev: bool, optional
    :param phase: The (complex) phase convention to use in the maximum alignment of the filter. Either of ['1/4', 'argmax']. See explanation below. Defaults to '1/4'.
    :type phase: str, optional

    **Baseline model basis functions:**

    +------------+--------------------------------+----------------------------+
    | ``basis``  | ``bspec``                      | Description                |
    +============+================================+============================+
    | 'poly'     | 0, 1, 2, ...                   | Polynomial degree,         |
    |            |                                | e.g. 3 -> cubic baseline.  |
    +------------+--------------------------------+----------------------------+
    | 'exp'      | [tau1, tau2, ...]              | List of decay constants to |
    |            |                                | consider in the model in   |
    |            |                                | **seconds**.               |
    +------------+--------------------------------+----------------------------+
    | 'exp_dtau' | [tau1, tau2, ...]              | Same as 'exp' but allows   |
    |            |                                | for first order deviations |
    |            |                                | of the time constants from |
    |            |                                | the given values.          |
    +------------+--------------------------------+----------------------------+
    | 'custom'   | [b1, b2, b3, ...]              | List of arrays which       |
    |            |                                | constitute the basis.      |
    +------------+--------------------------------+----------------------------+

    **Phase argument:**

    The complex phase (``np.exp(-1j*omega*phi)``, where ``omega=2*np.pi*np.fft.rfftfreq(sev.shape[-1])``) of a filter determines its lag in the time domain. Traditionally, it is chosen such that (in the time domain) the filter output peaks at the maximum position of a pulse. This is achieved with ``phi=np.argmax(sev)``. 

    For finite record windows, the chosen lag acts **circularly**. You can picture it like moving the SEV in the time domain sample by sample. Samples that fall outside the record window enter again on the other side. For a flat baseline model (which is trivially periodic) and SEVs that decay completely within the record window such shifts usually don't cause problems. **However**, if you describe the baseline using non-constant functions (e.g. polynomials) and/or your SEV is non-zero at the end of the record window, this wrap-around is an issue.

    There are two cases to consider::
        
        If you just Fourier transform a voltage trace, then multiply it by the filter in frequency domain, then **evaluate the result at a fixed sample**, the phase is irrelevant **as long as** the sample where you evaluate it matches the chosen phase. E.g. if you choose ``phase='argmax'`` and you evaluate the result at ``np.argmax(sev)``, you're good. Likewise, choosing ``phase='1/4`` and evaluating at ``sev.shape[-1]//4`` is fine. 

        If you want to **slide** the filter (like e.g. in :func:`cait.versatile.trigger_of` or :func:`cait.DataHandler.trigger_of`), the phase has to be zero. Otherwise the wrap-around would spoil the result. For backwards compatibility with traditional OFs (phase aligned with argmax of SEV, usually very close to 1/4th of the record window), those sliding trigger functions *automatically account for a phase of 1/4th of the record window*! This means that if you intend to use the filter for a sliding trigger, ``phase='1/4'`` has to be set when constructing the filter such that it is compensated in the trigger function and the resulting phase is zero again (I agree that this appears cumbersome and confusing, but while implementing this seemed like the best tradeoff between usability, clarity, and backwards compatibility).

    Trivially, if ``np.argmax(sev) == sev.shape[-1]//4``, phase arguments '1/4' and 'argmax' are equivalent.

    +-----------+--------------------------+-----------------------------------+
    | ``phase`` | Use case                 | Description                       |
    +===========+==========================+===================================+
    | '1/4'     | ``vai.trigger_of``       | Lag such that it is automatically |
    |           | ``dh.trigger_of``        | compensated by sliding trigger    |
    |           |                          | functions. Important when         |
    |           |                          | triggering with non-constant      |
    |           |                          | baselines and SEVs that are non-  |
    |           |                          | zero at the end of the window.    |
    +-----------+--------------------------+-----------------------------------+
    | 'argmax'  | ``vai.OptimumFiltering`` | Lag such that filter output       |
    |           | ``dh.apply_of``          | peaks at SEV maximum. Traditional |
    |           |                          | phase choice. Use for filter      |
    |           |                          | output evaluation at fixed sample.|
    +-----------+--------------------------+-----------------------------------+
    """
    _supported_bl_models = _supported_bl_models
    _supported_phase_args = ["1/4", "argmax"]

    def __init__(
            self, 
            sev: SEV, 
            nps: NPS,
            *,
            basis: str = "poly",
            bspec: Union[int, Tuple[List[float], bool], List[List[float]]] = 3,
            bcc: np.ndarray = None,
            window_sev: bool = False,
            phase: str = "1/4",
            ):
        # Input validation
        if not isinstance(sev, SEV):
            raise TypeError(f"Input argument 'sev' has to be of type 'SEV', not {type(sev).__name__}.")
        if not isinstance(nps, NPS):
            raise TypeError(f"Input argument 'nps' has to be of type 'NPS', not {type(nps).__name__}.")

        if sev.ndim > 1:
            raise ValueError(f"Input argument 'sev' has to be 1d. Got {sev.ndim}d.")
        if nps.shape != (sev.shape[-1]//2 + 1,):
            raise ValueError(f"Input argument 'nps' has to have shape (N//2+1,) for input argument 'sev' of shape (N,).")
        if nps.dt_us != sev.dt_us:
            raise ValueError(f"Input arguments 'sev' and 'nps' must have the same timebase (dt_us). Got {sev.dt_us} and {nps.dt_us}.")
        
        if not isinstance(phase, str):
            raise TypeError(f"Input argument 'phase' has to be of type 'str', not {type(phase).__name__}.")
        if phase.lower() not in self._supported_phase_args:
            raise ValueError(f"Input argument 'phase' must be one of {self._supported_phase_args}, not '{phase}'.")
        if np.argmax(sev) != sev.shape[-1]//4:
            warnings.warn(f"The maximum index of the provided 'sev' ({np.argmax(sev)}) is {ai.styles.txt_fmt('not aligned', style='bold')} with 1/4th of the record window ({sev.shape[-1]//4})! {ai.styles.txt_fmt('Make sure to use the appropriate ''phase'' argument!', style='bold')} Offset: {sev.shape[-1]//4 - np.argmax(sev)} samples.\n\nFor regular OFs (which assume a flat baseline model) and in cases where the SEV decays completely within the record window, the difference between phase arguments '1/4' and 'argmax' is negligible.\n{ai.styles.txt_fmt('HOWEVER, for non-constant baseline models and/or SEVs which are non-zero at the end of the record window', style='bold')}, this offset can lead to non-optimal or wrong results if 'phase' is not used appropriately.\n\nUse phase='1/4' if you intend to use the filter for triggering (using vai.trigger_of or dh.trigger_of) and phase='argmax' if you plan to multiply the filter in the frequency domain and evaluate it at the maximum position of the SEV. See also docstring for more information.\n\nIf you are annoyed by this warning because it clutters your Jupyter cell outputs {ai.styles.txt_fmt('and you know what you are doing', style='bold')}, you can just execute cells twice. Usually, Jupyter is configured to only show a warning once per cell.\n\n")

        B = construct_basis(sev=sev, basis=basis, bspec=bspec)
        K = len(B)

        if bcc is not None and np.array(bcc).shape != (K, K):
            raise ValueError(f"Input argument 'bcc' must have shape (K, K) where K is the number of basis components. Got shape {np.array(bcc).shape} and {K=}.")
        
        bcc_inv = np.zeros((K, K)) 
        if bcc is not None:
            bcc = np.array(bcc)
            # If any of the diagonal entries of bcc is infinite (i.e. infinite
            # variance <=> no constraint), we want the corresponding block
            # of the matrix to invert to zero entries. Below, we do this safely.
            constrained_coeffs = np.diag(bcc) != np.inf
            mask = constrained_coeffs[:, None]*constrained_coeffs[None, :]
            # Number of coefficients with finite variance
            n = np.sum(constrained_coeffs)

            bcc_inv[mask] = sp.linalg.solve(
                bcc[mask].reshape((n, n)), 
                np.eye(n),
                assume_a="her",
                ).flatten()
        
        # We now have a list of basis functions B. Next, we calculate
        # the corresponding frequency representations and the Gram matrix.
        dt = sev.dt_us*1e-6
        B_tilde = [np.fft.rfft(b) for b in B]
        G = gram_matrix(B_tilde=B_tilde, nps=nps, dt=dt)

        G_inv = sp.linalg.solve(G + bcc_inv, np.eye(K), assume_a="her")

        # Project basis out of template.
        # (Removing mean from SEV improves numerical stability)
        sev_tilde = np.fft.rfft(sev - np.mean(sev))
        sev_tilde_eff = np.fft.rfft(sev - np.mean(sev))

        tw = vai.TukeyWindow()
        sev_tilde_win = np.fft.rfft(tw(sev) - np.mean(tw(sev)))
        sev_tilde_eff_win = np.fft.rfft(tw(sev) - np.mean(tw(sev)))

        for j in range(K):
            for l in range(K):
                sev_tilde_eff -= inner_product(
                    sev_tilde, 
                    B_tilde[j], 
                    nps, 
                    dt,
                )*G_inv[j, l]*B_tilde[l]
                sev_tilde_eff_win -= inner_product(
                    sev_tilde_win, 
                    B_tilde[j], 
                    nps, 
                    dt,
                )*G_inv[j, l]*B_tilde[l]
        
        # Filter variances and kernel definition
        if window_sev:
            # Window loss
            kappa = inner_product(
                sev_tilde_eff_win, sev_tilde_win, nps, dt
                )/inner_product(
                    sev_tilde_eff_win, sev_tilde, nps, dt
                    )
            kappa_optimal = inner_product(
                sev_tilde_win, sev_tilde_win, nps, dt
                )/inner_product(
                    sev_tilde_win, sev_tilde, nps, dt
                    )
            
            var = 1/inner_product(sev_tilde_eff_win, sev_tilde_win, nps, dt)
            var_optimal = 1/inner_product(sev_tilde_win, sev_tilde_win, nps, dt)
        
            self._var, self._var_optimal = kappa**2 * var, kappa_optimal**2 * var_optimal
            h_tilde = 2 * kappa * dt * var * sev_tilde_eff_win / nps
        else:
            self._var = 1/inner_product(sev_tilde_eff, sev_tilde, nps, dt)
            self._var_optimal = 1/inner_product(sev_tilde, sev_tilde, nps, dt)
            h_tilde = 2 * dt * self._var * sev_tilde_eff / nps

        omega = 2 * np.pi * np.fft.rfftfreq(sev.shape[-1])
        if phase.lower() == "argmax":
            # Multiply phase such that the filtered maximum and the pulse
            # maximum are aligned (convention also for regular OF).
            phi = np.argmax(sev)
        elif phase.lower() == "1/4":
            # Multiply phase such that the it is automatically compensated
            # by sliding filter functions.
            phi = sev.shape[-1]/4

        super().__init__(h_tilde.conj()*np.exp(-1j * phi * omega), dt_us=sev.dt_us)

    def __repr__(self):
        s = super().__repr__().split(")")
        s[-2] += f", filter_var={self.filter_var:.3g}, filter_var_opt={self.filter_var_opt:.3g}"
        return ")".join(s)

    @classmethod
    def from_file(cls, fname: str, src_dir: str = ""):
        """
        Construct GOF from xy-file.

        :param fname: Filename to look for (without file-extension).
        :type fname: str
        :param out_dir: Directory to look in. Defaults to '' which means searching current directory.
        :type out_dir: str, optional

        :return: Instance of GOF.
        :rtype: GOF
        """
        # First load it as a regular OF ...
        of = OF.from_file(fname=fname, src_dir=src_dir)

        # ... then construct the GOF:
        # Placeholder GOF (GOF cannot be constructed from an array
        # directly as OF can)
        gof = cls(
            sev=SEV(np.linspace(0, 1, 16), dt_us=of.dt_us), 
            nps=NPS(np.ones(9), dt_us=of.dt_us), 
            basis="custom", 
            bspec=[np.ones(16)]
        )
        # Replace internals with correct data
        super(cls, gof).__init__(of, dt_us=of.dt_us)
        
        # Set remaining properties
        with open(os.path.join(src_dir, fname + ".xy"), "r") as f:
            first_line = f.readline()

        header = re.findall(r"\{.*\}", first_line)
        info_dict = json.loads(header[0])
        
        gof._var = info_dict["filter_var"]
        gof._var_optimal = info_dict["filter_var_opt"]

        return gof

    def to_file(self, fname: str, out_dir: str = "", info_str: str = ""):
        """
        Write GOF to xy-file.

        :param fname: Filename to use (without file-extension).
        :type fname: str
        :param out_dir: Directory to write to. Defaults to '' which means writing to current directory. 
        :type out_dir: str, optional
        :param info_str: An info string to be saved as a description of the GOF in the header of the file. Defaults to '', i.e. no info string.
        :type info_str: str, optional
        """
        # First save it as a regular OF ...
        super().to_file(fname=fname, out_dir=out_dir)

        # ... then replace the header
        new_header = (
            "GOF "
            + json.dumps(
                {
                    "cait": ai.__version__,
                    "dt_us": self.dt_us,
                    "record_length": 2 * (self._of.shape[-1] - 1),
                    "n_ch": self._n_channels,
                    "filter_var": self.filter_var,
                    "filter_var_opt": self.filter_var_opt,
                    **({"info": str(info_str)} if info_str else {})
                }
            )
            + "\n"
        )

        with open(os.path.join(out_dir, fname + ".xy"), "r+") as f:
            lines = f.readlines()
            lines[0] = new_header
            f.seek(0)
            f.writelines(lines)
            f.truncate()

    @property
    def filter_var(self):
        """Theoretical variance of the filter's amplitude estimate."""
        return self._var
    
    @property
    def filter_var_opt(self):
        """Theoretical variance of a filter's amplitude estimate on flat baselines (i.e. if no performance is lost to baseline compensation)."""
        return self._var_optimal