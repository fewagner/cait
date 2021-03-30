
import math
import numpy as np
import scipy.stats
import scipy.integrate
import numba
from ._bandfunctions import *


def reduceparvalues(parvalues, fixedvalues):
    """
    Return only the non-fixed parameter vector.
    """
    parvaluesred = np.array([])
    for i in np.arange(parvalues.shape[0]):
        if fixedvalues[i] == 0:
            parvaluesred = np.append(parvaluesred, parvalues[i])
    return parvaluesred


@numba.njit
def expandparvalues(parvalues, parvaluesred, fixedvalues):
    """
    Expands the reduced parameter values to the full parameter vector.
    """
    n = 0
    for i in np.arange(parvalues.shape[0]):
        if fixedvalues[i] == 0:
            parvalues[i] = parvaluesred[i - n]
        else:
            n = n + 1
    return parvalues


@numba.njit
def boundscheck(pars, lbounds,
                ubounds):
    """
    Parameter check, if parameter is out of bounds, return a large positive value.
    """
    retval = 0.0
    for i in np.arange(len(pars)):
        if pars[i] < lbounds[i]:
            retval += 1.0e100 - 1.0e100 * (pars[i] - lbounds[i])
        if pars[i] > ubounds[i]:
            retval += 1.0e100 - 1.0e100 * (ubounds[i] - pars[i])
    return retval


@numba.njit
def evaluation(i, xy, cuteffarr, roi, nn, ng, nb, ni, peb, plr, pel, ppr, pes, ples, plem, pgb, eps, kgd, thr, pnb, pns,
               pgs, pbs, pib, pis):
    """
    Returns the likelihood and integral over light for given data.
    """
    prob = 0.0
    iol = 0.0
    emean = meane(xy[i, 0], peb)
    ennmean = meanenn(xy[i, 0], peb)
    eslope = slopee(xy[i, 0], peb)
    ebandressq = bressq(xy[i, 0], emean, ppr, plr, thr, eslope)
    ebandres = np.sqrt(ebandressq)
    spec = pol1(xy[i, 0], pes)
    prob += spec * probability(xy[i, 1], emean, ebandres)
    iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], emean, ebandres)

    leebandres = np.sqrt(bressq(xy[i, 0], plem, ppr, plr, thr, 0.0))
    spec = expspec(xy[i, 0], ples)
    prob += spec * probability(xy[i, 1], plem, leebandres)
    iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], plem, leebandres)

    prob += probexli(xy[i, 0], xy[i, 1], pel, emean, ebandres, ebandressq)
    iol += exliintoverlight(xy[i, 0], xy[i, 0] * roi[2], xy[i, 0] * roi[3], pel, emean, ebandres, ebandressq)

    for j in np.arange(nn):
        nmean = meann(xy[i, 0], pnb[j, :], eps, ennmean)
        nslope = slopen(xy[i, 0], pnb[j, :], peb, eps)
        nbandres = np.sqrt(bressq(xy[i, 0], nmean, ppr, plr, thr, nslope))
        spec = expspec(xy[i, 0], pns[j, :])
        prob += spec * probability(xy[i, 1], nmean, nbandres)
        iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], nmean, nbandres)

    gcal = True
    gmean = 0.0
    gslope = 0.0
    gbandres = 0.0
    for j in np.arange(ng):
        pres = np.sqrt(pressq(pgs[j, 1], ppr, thr))
        if pgs[j, 1] - 6 * pres <= xy[i, 0] <= pgs[j, 1] + 6 * pres:
            if gcal:
                gmean = meang(xy[i, 0], pgb, peb)
                gslope = slopeg(xy[i, 0], pgb, peb)
                gbandres = np.sqrt(bressq(xy[i, 0], gmean, ppr, plr, thr, gslope))
                gcal = False
            spec = peak(xy[i, 0], pgs[j, :], pres)
            prob += spec * probability(xy[i, 1], gmean, gbandres)
            iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], gmean, gbandres)

    for j in np.arange(nb):
        preslower = np.sqrt(pressq(pbs[j, 1], ppr, thr))
        decaypoint = pbs[j, 1] + pbs[j, 2]
        presupper = np.sqrt(pressq(decaypoint, ppr, thr))
        if pbs[j, 1] - 6 * preslower <= xy[i, 0] <= pbs[j, 1]:
            gmean = meang(pbs[j, 1], pgb, peb)
            gslope = slopeg(pbs[j, 1], pgb, peb)
            gbandres = np.sqrt(bressq(pbs[j, 1], gmean, ppr, plr, thr, gslope))
            spec = gbspec(xy[i, 0], pbs[j, :], preslower)
            prob += spec * probability(xy[i, 1], gmean, gbandres)
            iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], gmean, gbandres)
        elif xy[i, 0] <= decaypoint:
            pres = np.sqrt(pressq(xy[i, 0], ppr, thr))
            bmean = meanb(xy[i, 0], pbs[j, 1], peb, pgb)
            bslope = slopeb(xy[i, 0], pbs[j, 1], peb)
            bbandres = np.sqrt(bressq(xy[i, 0], bmean, ppr, plr, thr, bslope))
            spec = gbspec(xy[i, 0], pbs[j, :], pres)
            prob += spec * probability(xy[i, 1], bmean, bbandres)
            iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], bmean, bbandres)
        elif xy[i, 0] <= decaypoint + 6 * presupper:
            bmean = meanb(decaypoint, pbs[j, 1], peb, pgb)
            bslope = slopeb(decaypoint, pbs[j, 1], peb)
            bbandres = np.sqrt(bressq(xy[i, 0], bmean, ppr, plr, thr, bslope))
            spec = gbspec(xy[i, 0], pbs[j, :], presupper)
            prob += spec * probability(xy[i, 1], bmean, bbandres)
            iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], bmean, bbandres)

    for j in np.arange(ni):
        preslower = np.sqrt(pressq(pis[j, 0], ppr, thr))
        presupper = np.sqrt(pressq(pis[j, 1], ppr, thr))
        if pis[j, 0] - 6 * preslower <= xy[i, 0] <= pis[j, 0]:
            emean = meane(pis[j, 0], peb)
            eslope = slopee(pis[j, 0], peb)
            ebandres = np.sqrt(bressq(pis[j, 0], emean, ppr, plr, thr, eslope))
            spec = ielspec(xy[i, 0], pis[j, :], preslower, presupper)
            prob += spec * probability(xy[i, 1], emean, ebandres)
            iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], emean, ebandres)
        elif xy[i, 0] <= pis[j, 1]:
            iemean = meanie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
            ieslope = slopeie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
            iebandres = np.sqrt(bressq(xy[i, 0], iemean, ppr, plr, thr, ieslope))
            spec = ielspec(xy[i, 0], pis[j, :], preslower, presupper)
            prob += spec * probability(xy[i, 1], iemean, iebandres)
            iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], iemean, iebandres)
        elif xy[i, 0] <= pis[j, 1] + 6 * presupper:
            iemean = meanie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
            ieslope = slopeie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
            iebandres = np.sqrt(bressq(xy[i, 0], iemean, ppr, plr, thr, ieslope))
            spec = ielspec(xy[i, 0], pis[j, :], preslower, presupper)
            prob += spec * probability(xy[i, 1], iemean, iebandres)
            iol += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], iemean, iebandres)
    prob = np.log(cuteffarr[i] * prob)
    iol *= cuteffarr[i]
    return prob, iol


def loglik(p, lbounds, ubounds, xy, cuteffarr, roi, nn, ng, nb, ni):
    """
    Likelihood function.
    """
    retval = 0.0
    retval += boundscheck(p, lbounds, ubounds)
    if retval > 0.0:
        return retval
    peb = p[0:4]  # electron band parameters; 0 = L0; 1 = L1; 2 = np_fract; 3 = np_decay
    plr = p[4:7]  # light resolution parameters; 0 = sigma_l0; 1 = S1; 2 = S2
    pel = p[7:10]  # excess light parameters; 0 = el_amp; 1 = el_decay; 2 = el_width
    ppr = p[10:12]  # phonon resolution parameters; 0 = sigma_p0; 1 = sigma_p1
    pes = p[12:14]  # electron spectrum parameters; 0 = E_p0; 1 = E_p1
    ples = p[14:16]  # low energy excess spectrum parameters; 0 = E_fr; 1 = E_dc
    plem = p[16]  # low energy excess mean parameter; p = L_lee
    pgb = p[17:19]  # gamma band parameters; 0 = QF_y; 1 = QF_ye
    eps = p[19]  # epsilon (common nuclear recoil quenching) parameter;
    kgd = p[20]  # exposure parameter;
    thr = p[21]  # threshold parameter;
    pnb = np.zeros((nn, 3))  # neutron band parameters; 0 = QF; 1 = es_f; 1 = es_lb
    pns = np.zeros((nn, 2))  # neutron spectrum parameters; 0 = nc_p0; 1 = nc_p1
    startindex = 22
    for j in np.arange(nn):
        pnb[j, :] = p[startindex:startindex + 3]  # neutron band parameters; 0 = QF; 1 = es_f; 1 = es_lb
        pns[j, :] = p[startindex + 3:startindex + 5]  # neutron spectrum parameters; 0 = nc_p0; 1 = nc_p1
        startindex += 5

    pgs = np.zeros((ng, 2))  # gamma peak parameters; 0 = FG_C; 1 = FG_M
    for j in np.arange(ng):
        pgs[j, :] = p[startindex:startindex + 2]
        startindex += 2

    pbs = np.zeros((nb, 3))  # beta peak parameters; 0 = FB_C; 1 = FB_M; 2 = FB_D
    for j in np.arange(nb):
        pbs[j, :] = p[startindex:startindex + 3]
        startindex += 3

    pib = np.zeros(ni, dtype=int)  # material which the inelastics correspond to; IE_M
    pis = np.zeros((ni, 4))  # inelastic band parameters; 0 = IE_S; 1 = IE_E; 2 = IE_p0; 3 = IE_p1
    for j in np.arange(ni):
        pib[j] = math.floor(p[startindex]) % nn
        pis[j, :] = p[startindex + 1:startindex + 5]
        startindex += 5

    iofarr = np.zeros(xy.shape[0])
    parr = np.zeros(xy.shape[0])
    for i in np.arange(xy.shape[0]):
        prob = 0.0
        emean = meane(xy[i, 0], peb)
        ennmean = meanenn(xy[i, 0], peb)
        eslope = slopee(xy[i, 0], peb)
        ebandressq = bressq(xy[i, 0], emean, ppr, plr, thr, eslope)
        ebandres = np.sqrt(ebandressq)
        spec = pol1(xy[i, 0], pes)
        prob += spec * probability(xy[i, 1], emean, ebandres)
        iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], emean, ebandres)

        leebandres = np.sqrt(bressq(xy[i, 0], plem, ppr, plr, thr, 0.0))
        spec = expspec(xy[i, 0], ples)
        prob += spec * probability(xy[i, 1], plem, leebandres)
        iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], plem, leebandres)

        prob += probexli(xy[i, 0], xy[i, 1], pel, emean, ebandres, ebandressq)
        iofarr[i] += exliintoverlight(xy[i, 0], xy[i, 0] * roi[2], xy[i, 0] * roi[3], pel, emean, ebandres, ebandressq)

        for j in np.arange(nn):
            nmean = meann(xy[i, 0], pnb[j, :], eps, ennmean)
            nslope = slopen(xy[i, 0], pnb[j, :], peb, eps)
            nbandres = np.sqrt(bressq(xy[i, 0], nmean, ppr, plr, thr, nslope))
            spec = expspec(xy[i, 0], pns[j, :])
            prob += spec * probability(xy[i, 1], nmean, nbandres)
            iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], nmean, nbandres)

        gcal = True
        gmean = 0.0
        gslope = 0.0
        gbandres = 0.0
        for j in np.arange(ng):
            pres = np.sqrt(pressq(pgs[j, 1], ppr, thr))
            if pgs[j, 1] - 6 * pres <= xy[i, 0] <= pgs[j, 1] + 6 * pres:
                if gcal:
                    gmean = meang(xy[i, 0], pgb, peb)
                    gslope = slopeg(xy[i, 0], pgb, peb)
                    gbandres = np.sqrt(bressq(xy[i, 0], gmean, ppr, plr, thr, gslope))
                    gcal = False
                spec = peak(xy[i, 0], pgs[j, :], pres)
                prob += spec * probability(xy[i, 1], gmean, gbandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], gmean, gbandres)

        for j in np.arange(nb):
            preslower = np.sqrt(pressq(pbs[j, 1], ppr, thr))
            decaypoint = pbs[j, 1] + pbs[j, 2]
            presupper = np.sqrt(pressq(decaypoint, ppr, thr))
            if pbs[j, 1] - 6 * preslower <= xy[i, 0] <= pbs[j, 1]:
                gmean = meang(pbs[j, 1], pgb, peb)
                gslope = slopeg(pbs[j, 1], pgb, peb)
                gbandres = np.sqrt(bressq(pbs[j, 1], gmean, ppr, plr, thr, gslope))
                spec = gbspec(xy[i, 0], pbs[j, :], preslower)
                prob += spec * probability(xy[i, 1], gmean, gbandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], gmean, gbandres)
            elif xy[i, 0] <= decaypoint:
                pres = np.sqrt(pressq(xy[i, 0], ppr, thr))
                bmean = meanb(xy[i, 0], pbs[j, 1], peb, pgb)
                bslope = slopeb(xy[i, 0], pbs[j, 1], peb)
                bbandres = np.sqrt(bressq(xy[i, 0], bmean, ppr, plr, thr, bslope))
                spec = gbspec(xy[i, 0], pbs[j, :], pres)
                prob += spec * probability(xy[i, 1], bmean, bbandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], bmean, bbandres)
            elif xy[i, 0] <= decaypoint + 6 * presupper:
                bmean = meanb(decaypoint, pbs[j, 1], peb, pgb)
                bslope = slopeb(decaypoint, pbs[j, 1], peb)
                bbandres = np.sqrt(bressq(xy[i, 0], bmean, ppr, plr, thr, bslope))
                spec = gbspec(xy[i, 0], pbs[j, :], presupper)
                prob += spec * probability(xy[i, 1], bmean, bbandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], bmean, bbandres)

        for j in np.arange(ni):
            preslower = np.sqrt(pressq(pis[j, 0], ppr, thr))
            presupper = np.sqrt(pressq(pis[j, 1], ppr, thr))
            if pis[j, 0] - 6 * preslower <= xy[i, 0] <= pis[j, 0]:
                emean = meane(pis[j, 0], peb)
                eslope = slopee(pis[j, 0], peb)
                ebandres = np.sqrt(bressq(pis[j, 0], emean, ppr, plr, thr, eslope))
                spec = ielspec(xy[i, 0], pis[j, :], preslower, presupper)
                prob += spec * probability(xy[i, 1], emean, ebandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], emean, ebandres)
            elif xy[i, 0] <= pis[j, 1]:
                iemean = meanie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
                ieslope = slopeie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
                iebandres = np.sqrt(bressq(xy[i, 0], iemean, ppr, plr, thr, ieslope))
                spec = ielspec(xy[i, 0], pis[j, :], preslower, presupper)
                prob += spec * probability(xy[i, 1], iemean, iebandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], iemean, iebandres)
            elif xy[i, 0] <= pis[j, 1] + 6 * presupper:
                iemean = meanie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
                ieslope = slopeie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
                iebandres = np.sqrt(bressq(xy[i, 0], iemean, ppr, plr, thr, ieslope))
                spec = ielspec(xy[i, 0], pis[j, :], preslower, presupper)
                prob += spec * probability(xy[i, 1], iemean, iebandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], iemean, iebandres)
        retval -= np.log(cuteffarr[i] * prob)
        iofarr[i] *= cuteffarr[i]

    uniquex, uniquei = np.unique(xy[:, 0], return_index=True, axis=None)
    ival = scipy.integrate.simpson(iofarr[uniquei], uniquex)
    # print(ival)
    retval += ival
    # print(retval)
    return retval


@numba.njit
def logliknoint(p, lbounds, ubounds, xy, cuteffarr, roi, nn, ng, nb, ni):
    """
    Likelihood function with pool multiprocessing and numba compatible.
    """
    retval = 0.0
    retval += boundscheck(p, lbounds, ubounds)
    if retval > 0.0:
        return retval, np.array([0.0]), False
    peb = p[0:4]  # electron band parameters; 0 = L0; 1 = L1; 2 = np_fract; 3 = np_decay
    plr = p[4:7]  # light resolution parameters; 0 = sigma_l0; 1 = S1; 2 = S2
    pel = p[7:10]  # excess light parameters; 0 = el_amp; 1 = el_decay; 2 = el_width
    ppr = p[10:12]  # phonon resolution parameters; 0 = sigma_p0; 1 = sigma_p1
    pes = p[12:14]  # electron spectrum parameters; 0 = E_p0; 1 = E_p1
    ples = p[14:16]  # low energy excess spectrum parameters; 0 = E_fr; 1 = E_dc
    plem = p[16]  # low energy excess mean parameter; p = L_lee
    pgb = p[17:19]  # gamma band parameters; 0 = QF_y; 1 = QF_ye
    eps = p[19]  # epsilon (common nuclear recoil quenching) parameter;
    kgd = p[20]  # exposure parameter;
    thr = p[21]  # threshold parameter;
    pnb = np.zeros((nn, 3))  # neutron band parameters; 0 = QF; 1 = es_f; 1 = es_lb
    pns = np.zeros((nn, 2))  # neutron spectrum parameters; 0 = nc_p0; 1 = nc_p1
    startindex = 22
    for j in np.arange(nn):
        pnb[j, :] = p[startindex:startindex + 3]  # neutron band parameters; 0 = QF; 1 = es_f; 1 = es_lb
        pns[j, :] = p[startindex + 3:startindex + 5]  # neutron spectrum parameters; 0 = nc_p0; 1 = nc_p1
        startindex += 5

    pgs = np.zeros((ng, 2))  # gamma peak parameters; 0 = FG_C; 1 = FG_M
    for j in np.arange(ng):
        pgs[j, :] = p[startindex:startindex + 2]
        startindex += 2

    pbs = np.zeros((nb, 3))  # beta peak parameters; 0 = FB_C; 1 = FB_M; 2 = FB_D
    for j in np.arange(nb):
        pbs[j, :] = p[startindex:startindex + 3]
        startindex += 3

    pib = np.zeros(ni, dtype=numba.int64)  # material which the inelastics correspond to; IE_M
    pis = np.zeros((ni, 4))  # inelastic band parameters; 0 = IE_S; 1 = IE_E; 2 = IE_p0; 3 = IE_p1
    for j in np.arange(ni):
        pib[j] = math.floor(p[startindex]) % nn
        pis[j, :] = p[startindex + 1:startindex + 5]
        startindex += 5

    iofarr = np.zeros(xy.shape[0])
    parr = np.zeros(xy.shape[0])
    for i in np.arange(xy.shape[0]):
        prob = 0.0
        emean = meane(xy[i, 0], peb)
        ennmean = meanenn(xy[i, 0], peb)
        eslope = slopee(xy[i, 0], peb)
        ebandressq = bressq(xy[i, 0], emean, ppr, plr, thr, eslope)
        ebandres = np.sqrt(ebandressq)
        spec = pol1(xy[i, 0], pes)
        prob += spec * probability(xy[i, 1], emean, ebandres)
        iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], emean, ebandres)

        leebandres = np.sqrt(bressq(xy[i, 0], plem, ppr, plr, thr, 0.0))
        spec = expspec(xy[i, 0], ples)
        prob += spec * probability(xy[i, 1], plem, leebandres)
        iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], plem, leebandres)

        prob += probexli(xy[i, 0], xy[i, 1], pel, emean, ebandres, ebandressq)
        iofarr[i] += exliintoverlight(xy[i, 0], xy[i, 0] * roi[2], xy[i, 0] * roi[3], pel, emean, ebandres, ebandressq)

        for j in np.arange(nn):
            nmean = meann(xy[i, 0], pnb[j, :], eps, ennmean)
            nslope = slopen(xy[i, 0], pnb[j, :], peb, eps)
            nbandres = np.sqrt(bressq(xy[i, 0], nmean, ppr, plr, thr, nslope))
            spec = expspec(xy[i, 0], pns[j, :])
            prob += spec * probability(xy[i, 1], nmean, nbandres)
            iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], nmean, nbandres)

        gcal = True
        gmean = 0.0
        gslope = 0.0
        gbandres = 0.0
        for j in np.arange(ng):
            pres = np.sqrt(pressq(pgs[j, 1], ppr, thr))
            if pgs[j, 1] - 6 * pres <= xy[i, 0] <= pgs[j, 1] + 6 * pres:
                if gcal:
                    gmean = meang(xy[i, 0], pgb, peb)
                    gslope = slopeg(xy[i, 0], pgb, peb)
                    gbandres = np.sqrt(bressq(xy[i, 0], gmean, ppr, plr, thr, gslope))
                    gcal = False
                spec = peak(xy[i, 0], pgs[j, :], pres)
                prob += spec * probability(xy[i, 1], gmean, gbandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], gmean, gbandres)

        for j in np.arange(nb):
            preslower = np.sqrt(pressq(pbs[j, 1], ppr, thr))
            decaypoint = pbs[j, 1] + pbs[j, 2]
            presupper = np.sqrt(pressq(decaypoint, ppr, thr))
            if pbs[j, 1] - 6 * preslower <= xy[i, 0] <= pbs[j, 1]:
                gmean = meang(pbs[j, 1], pgb, peb)
                gslope = slopeg(pbs[j, 1], pgb, peb)
                gbandres = np.sqrt(bressq(pbs[j, 1], gmean, ppr, plr, thr, gslope))
                spec = gbspec(xy[i, 0], pbs[j, :], preslower)
                prob += spec * probability(xy[i, 1], gmean, gbandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], gmean, gbandres)
            elif xy[i, 0] <= decaypoint:
                pres = np.sqrt(pressq(xy[i, 0], ppr, thr))
                bmean = meanb(xy[i, 0], pbs[j, 1], peb, pgb)
                bslope = slopeb(xy[i, 0], pbs[j, 1], peb)
                bbandres = np.sqrt(bressq(xy[i, 0], bmean, ppr, plr, thr, bslope))
                spec = gbspec(xy[i, 0], pbs[j, :], pres)
                prob += spec * probability(xy[i, 1], bmean, bbandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], bmean, bbandres)
            elif xy[i, 0] <= decaypoint + 6 * presupper:
                bmean = meanb(decaypoint, pbs[j, 1], peb, pgb)
                bslope = slopeb(decaypoint, pbs[j, 1], peb)
                bbandres = np.sqrt(bressq(xy[i, 0], bmean, ppr, plr, thr, bslope))
                spec = gbspec(xy[i, 0], pbs[j, :], presupper)
                prob += spec * probability(xy[i, 1], bmean, bbandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], bmean, bbandres)

        for j in np.arange(ni):
            preslower = np.sqrt(pressq(pis[j, 0], ppr, thr))
            presupper = np.sqrt(pressq(pis[j, 1], ppr, thr))
            if pis[j, 0] - 6 * preslower <= xy[i, 0] <= pis[j, 0]:
                emean = meane(pis[j, 0], peb)
                eslope = slopee(pis[j, 0], peb)
                ebandres = np.sqrt(bressq(pis[j, 0], emean, ppr, plr, thr, eslope))
                spec = ielspec(xy[i, 0], pis[j, :], preslower, presupper)
                prob += spec * probability(xy[i, 1], emean, ebandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], emean, ebandres)
            elif xy[i, 0] <= pis[j, 1]:
                iemean = meanie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
                ieslope = slopeie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
                iebandres = np.sqrt(bressq(xy[i, 0], iemean, ppr, plr, thr, ieslope))
                spec = ielspec(xy[i, 0], pis[j, :], preslower, presupper)
                prob += spec * probability(xy[i, 1], iemean, iebandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], iemean, iebandres)
            elif xy[i, 0] <= pis[j, 1] + 6 * presupper:
                iemean = meanie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
                ieslope = slopeie(xy[i, 0], pis[j, 0], peb, pnb[pib[j], :], eps)
                iebandres = np.sqrt(bressq(xy[i, 0], iemean, ppr, plr, thr, ieslope))
                spec = ielspec(xy[i, 0], pis[j, :], preslower, presupper)
                prob += spec * probability(xy[i, 1], iemean, iebandres)
                iofarr[i] += spec * intoverlight(xy[i, 0] * roi[2], xy[i, 0] * roi[3], iemean, iebandres)
        retval -= np.log(cuteffarr[i] * prob)
        iofarr[i] *= cuteffarr[i]

    return retval, iofarr, True


def loglikpool(p, lbounds, ubounds, xy, cuteffarr, roi, nn, ng, nb, ni):
    """
    Likelihood function with pool multiprocessing.
    """
    retval = 0.0
    retval += boundscheck(p, lbounds, ubounds)
    if retval > 0.0:
        return retval
    peb = p[0:4]  # electron band parameters; 0 = L0; 1 = L1; 2 = np_fract; 3 = np_decay
    plr = p[4:7]  # light resolution parameters; 0 = sigma_l0; 1 = S1; 2 = S2
    pel = p[7:10]  # excess light parameters; 0 = el_amp; 1 = el_decay; 2 = el_width
    ppr = p[10:12]  # phonon resolution parameters; 0 = sigma_p0; 1 = sigma_p1
    pes = p[12:14]  # electron spectrum parameters; 0 = E_p0; 1 = E_p1
    ples = p[14:16]  # low energy excess spectrum parameters; 0 = E_fr; 1 = E_dc
    plem = p[16]  # low energy excess mean parameter; p = L_lee
    pgb = p[17:19]  # gamma band parameters; 0 = QF_y; 1 = QF_ye
    eps = p[19]  # epsilon (common nuclear recoil quenching) parameter;
    kgd = p[20]  # exposure parameter;
    thr = p[21]  # threshold parameter;
    pnb = np.zeros((nn, 3))  # neutron band parameters; 0 = QF; 1 = es_f; 1 = es_lb
    pns = np.zeros((nn, 2))  # neutron spectrum parameters; 0 = nc_p0; 1 = nc_p1
    startindex = 22
    for j in np.arange(nn):
        pnb[j, :] = p[startindex:startindex + 3]  # neutron band parameters; 0 = QF; 1 = es_f; 1 = es_lb
        pns[j, :] = p[startindex + 3:startindex + 5]  # neutron spectrum parameters; 0 = nc_p0; 1 = nc_p1
        startindex += 5

    pgs = np.zeros((ng, 2))  # gamma peak parameters; 0 = FG_C; 1 = FG_M
    for j in np.arange(ng):
        pgs[j, :] = p[startindex:startindex + 2]
        startindex += 2

    pbs = np.zeros((nb, 3))  # beta peak parameters; 0 = FB_C; 1 = FB_M; 2 = FB_D
    for j in np.arange(nb):
        pbs[j, :] = p[startindex:startindex + 3]
        startindex += 3

    pib = np.zeros(ni, dtype=int)  # material which the inelastics correspond to; IE_M
    pis = np.zeros((ni, 4))  # inelastic band parameters; 0 = IE_S; 1 = IE_E; 2 = IE_p0; 3 = IE_p1
    for j in np.arange(ni):
        pib[j] = math.floor(p[startindex]) % nn
        pis[j, :] = p[startindex + 1:startindex + 5]
        startindex += 5

    iofarr = np.zeros(xy.shape[0])
    parr = np.zeros(xy.shape[0])
    # pool = multiprocessing.Pool(4)
    # prob, iol= zip(*pool.map(evaluation, np.arange(xy.shape[0])))
    for i in np.arange(xy.shape[0]):
        parr[i], iofarr[i] = evaluation(i, xy, cuteffarr, roi, nn, ng, nb, ni, peb, plr, pel, ppr, pes, ples, plem, pgb,
                                        eps, kgd, thr, pnb, pns, pgs, pbs, pib, pis)

    retval = -np.sum(parr)
    uniquex, uniquei = np.unique(xy[:, 0], return_index=True, axis=None)
    ival = scipy.integrate.simpson(iofarr[uniquei], uniquex)
    # print(ival)
    retval += ival
    # print(retval)
    return retval


def wrapper(p, parvalues, fixedvalues, lbounds, ubounds, xy, cuteffarr, roi, nn, ng, nb, ni):
    """
    Wrapper for the likelihood function to fill into minimizer.
    """
    parvalues = expandparvalues(parvalues, p, fixedvalues)
    return loglik(parvalues, lbounds, ubounds, xy, cuteffarr, roi, nn, ng, nb, ni)


def wrapperpool(p, parvalues, fixedvalues, lbounds, ubounds, xy, cuteffarr, roi, nn, ng, nb, ni):
    """
    Wrapper for the likelihood function to fill into minimizer, uses pool multiprocessing.
    """
    parvalues = expandparvalues(parvalues, p, fixedvalues)
    return loglikpool(parvalues, lbounds, ubounds, xy, cuteffarr, roi, nn, ng, nb, ni)


def wrappernoint(p, parvalues, fixedvalues, lbounds, ubounds, xy, cuteffarr, roi, nn, ng, nb, ni, info):
    """
    Wrapper for the likelihood function to fill into minimizer, uses pool multiprocessing, works with numba.
    """
    parvalues = expandparvalues(parvalues, p, fixedvalues)
    retval, iofarr, tf = logliknoint(parvalues, lbounds, ubounds, xy, cuteffarr, roi, nn, ng, nb, ni)
    if tf:
        uniquex, uniquei = np.unique(xy[:, 0], return_index=True, axis=None)
        retval += scipy.integrate.simpson(iofarr[uniquei], uniquex)

    # display information
    if 'Nfeval' in info:
        if info['Nfeval'] % 500 == 0:
            print('{}   {}'.format(info['Nfeval'], retval))
        info['Nfeval'] += 1

    return retval