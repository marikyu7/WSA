import numpy as np
import scipy
from multiSyncPy import synchrony_metrics as sm
from multiSyncPy import data_generation as dg
import random
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def symbolic_entropy(data, CI=True, alpha=0.05, n_resamplings = 50):
    """Computes entropy after mapping the signals to numbers representing 'low', 'medium' and 'high' values, and then concatenating these numbers (across the signals) to create a 'pattern' at each time step. The thresholds for deciding 'low', 'medium' and 'high' are terciles.

    Parameters
    ----------
    data: ndarray
        An array containing the time series of measurements, with the shape (number_signals, duration).
    CI: True
        if True returns confidence intervals for the estimation (obtained through bootstrapping)
    alpha: 0.05
        alpha value for the confidence interval (default: 0.05)
    n_resamplings: 100
        number of resemplings for the bootstrapping procedure (to build the confidence interval)

    Returns
    -------
    pattern_entropy: float
        The Shannon entropy of symbols found by mapping the input signals to 'low', 'medium' and 'high' and concatenating across signals.
    """

    data_terciles = np.apply_along_axis(sm.convert_to_terciles, 1, data)
    data_patterns = np.apply_along_axis(lambda x: "".join([str(int(y)) for y in x]), 0, data_terciles)

    # bootstrapping procedure
    if CI == True:
        sample_entropy = []
        for i in range(n_resamplings):
            y = random.choices(data_patterns.tolist(), k = data.shape[1])
            pattern_probabilities = np.unique(y, return_counts=True)[1] / len(y)
            entropy = -np.sum(pattern_probabilities * np.log(pattern_probabilities))
            sample_entropy.append(entropy)
        return mean_confidence_interval(sample_entropy, confidence=1-alpha)

    else:
        pattern_probabilities = np.unique(data_patterns, return_counts=True)[1] / data_patterns.shape[0]
        return -np.sum(pattern_probabilities * np.log(pattern_probabilities))

def sum_normalized_csd(data, CI = True, n_resamplings = 50, alpha = 0.05):
    """Returns a quantity, based on the cross-spectral density (CSD), similar to that of coherence_team() but which is less impacted by Gaussian noise.

    Parameters
    ----------
    data: ndarray
        An array containing the time series of measurements with shape (number_signals, duration).
    CI: True
        if True returns confidence intervals for the estimation (obtained through bootstrapping)
    alpha: 0.05
        alpha value for the confidence interval (default: 0.05)
    n_resamplings: 100
        number of resemplings for the bootstrapping procedure (to build the confidence interval)

    Returns
    -------
    aggregated_csd: float
        The sum-normalized CSD quantity.
    """

    ## Set nperseg to a reasonable value for shorter input lengths
    if data.shape[1] // 256 < 4:  ## Default value is 256

        nperseg = data.shape[1] // 4

    else:

        nperseg = None  ## Let scipy set it to default value

    csd_scores = []

    for i, x in enumerate(data):

        for j, y in enumerate(data):

            if i < j:
                csd_scores.append(
                    (np.abs(scipy.signal.csd(x, y, nperseg=nperseg)[1]) ** 2).sum() / (
                                scipy.signal.csd(x, x, nperseg=nperseg)[1] * scipy.signal.csd(y, y, nperseg=nperseg)[
                            1]).sum()
                )

    if CI == True:
        bootstrapped_coherence = []
        for n in range(n_resamplings):
            r = random.choices(csd_scores, k=len(csd_scores))
            bootstrapped_coherence.append(np.mean(r))

        return mean_confidence_interval(bootstrapped_coherence, confidence=1 - alpha)

    else:
        return np.mean(csd_scores)

def rho(phases, CI = True, n_resamplings = 50, alpha = 0.05):
    """Returns the quantity defined by Richardson et al. as 'rho' in "Measuring group synchrony: a cluster-phase method foranalyzing multivariate movement time-series:, doi: 10.3389/fphys.2012.00405.

    Parameters
    ----------
    phases: ndarray
        The phase time series (in radians) of the signals with the shape (number_signals, duration).
    CI: True
        if True returns confidence intervals for the estimation (obtained through bootstrapping)
    alpha: 0.05
        alpha value for the confidence interval (default: 0.05)

    Returns
    -------
    rho_group: ndarray
        The quantity rho averaged over time.
    """

    # Group level
    q_dash = np.exp(phases * 1j).mean(axis=0)
    q = np.arctan2(q_dash.imag, q_dash.real)
    # Individual level
    phi = phases - q
    phi_bar_dash = np.exp(phi * 1j).mean(axis=1)
    phi_bar = np.arctan2(phi_bar_dash.imag, phi_bar_dash.real)
    rho = np.abs(phi_bar_dash)
    # Group level
    rho_group_i = np.abs(np.exp((phi - phi_bar[:, None]) * 1j).mean(axis=0))

    if CI == True:
        bootstrapped_rho_group = []
        for n in range(n_resamplings):
            r = random.choices(rho_group_i, k=len(rho_group_i))
            bootstrapped_rho_group.append(np.mean(r))

        return mean_confidence_interval(bootstrapped_rho_group, alpha)

    else:
        rho_group = rho_group_i.mean()

        return rho_group


def signal():
    noise_1 = np.vstack((np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)])))
    noise_2 = np.vstack((np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)])))
    noise_3 = np.vstack((np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)])))
    noise_4 = np.vstack((np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)])))

    oscillators_1 = dg.kuramoto_data(np.array([0, 0, 0, 0, 0]), [20, 20, 20, 20, 20], 0.6, 0, 0.01, 20)
    oscillators_1[1] = oscillators_1[0] + 0.7
    oscillators_1[2] = oscillators_1[0] + 1.4
    oscillators_1[3] = oscillators_1[0] - 0.7
    oscillators_1[4] = oscillators_1[0] - 1.4

    oscillators_2 = dg.kuramoto_data(np.array([0, 0, 0, 0, 0]), [20, 20, 20, 20, 20], 0.6, 0, 0.01, 50)
    oscillators_2[1] = oscillators_2[0] + 0.7
    oscillators_2[2] = oscillators_2[0] + 1.4
    oscillators_2[3] = oscillators_2[0] - 0.7
    oscillators_2[4] = oscillators_2[0] - 1.4

    oscillators_3 = dg.kuramoto_data(np.array([0, 0, 0, 0, 0]), [20, 20, 20, 20, 20], 0.6, 0, 0.01, 70)
    oscillators_3[1] = oscillators_3[0] + 0.7
    oscillators_3[2] = oscillators_3[0] + 1.4
    oscillators_3[3] = oscillators_3[0] - 0.7
    oscillators_3[4] = oscillators_3[0] - 1.4

    oscillators_4 = dg.kuramoto_data(np.array([0, 0, 0, 0, 0]), [20, 20, 20, 20, 20], 0.6, 0, 0.01, 100)
    oscillators_4[1] = oscillators_4[0] + 0.7
    oscillators_4[2] = oscillators_4[0] + 1.4
    oscillators_4[3] = oscillators_4[0] - 0.7
    oscillators_4[4] = oscillators_4[0] - 1.4

    noise_rest = np.vstack(((
    np.array([random.gauss(0.0, 0.5) for i in range(200)]), np.array([random.gauss(0.0, 0.5) for i in range(200)]),
    np.array([random.gauss(0.0, 0.5) for i in range(200)]), np.array([random.gauss(0.0, 0.5) for i in range(200)]),
    np.array([random.gauss(0.0, 0.5) for i in range(200)]))))

    synthetic = np.concatenate(
        (noise_1, oscillators_1, noise_2, oscillators_2, noise_3, oscillators_3, noise_4, oscillators_4, noise_rest),
        axis=1)

    return synthetic

def signal_noised():
    kuramoto_args = {
        "K":0.6,
        "phases":np.array([0, 0, 0, 0, 0]),
        "omegas":[20,20,20,20,20],
        "alpha":0.95,
        "d_t":0.01,
        "length":100
    }

    kuramoto_test_data = dg.kuramoto_data(**kuramoto_args)

    noise_1 = np.vstack((np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)])))
    noise_2 = np.vstack((np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)])))
    noise_3 = np.vstack((np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)])))
    noise_4 = np.vstack((np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)]),
                         np.array([random.gauss(0.0, 0.5) for i in range(200)])))

    oscillators_1 = dg.kuramoto_data(np.array([0, 0, 0, 0, 0]), [20, 20, 20, 20, 20], 0.6, 1, 0.01, 20)
    oscillators_1[1] = oscillators_1[1] + 0.7
    oscillators_1[2] = oscillators_1[2] + 1.4
    oscillators_1[3] = oscillators_1[3] - 0.7
    oscillators_1[4] = oscillators_1[4] - 1.4

    oscillators_2 = dg.kuramoto_data(np.array([0, 0, 0, 0, 0]), [20, 20, 20, 20, 20], 0.6, 1, 0.01, 50)
    oscillators_2[1] = oscillators_2[1] + 0.7
    oscillators_2[2] = oscillators_2[2] + 1.4
    oscillators_2[3] = oscillators_2[3] - 0.7
    oscillators_2[4] = oscillators_2[4] - 1.4

    oscillators_3 = dg.kuramoto_data(np.array([0, 0, 0, 0, 0]), [20, 20, 20, 20, 20], 0.6, 1, 0.01, 70)
    oscillators_3[1] = oscillators_3[1] + 0.7
    oscillators_3[2] = oscillators_3[2] + 1.4
    oscillators_3[3] = oscillators_3[3] - 0.7
    oscillators_3[4] = oscillators_3[4] - 1.4

    oscillators_4 = dg.kuramoto_data(np.array([0, 0, 0, 0, 0]), [20, 20, 20, 20, 20], 0.6, 1, 0.01, 100)
    oscillators_4[1] = oscillators_4[1] + 0.7
    oscillators_4[2] = oscillators_4[2] + 1.4
    oscillators_4[3] = oscillators_4[3] - 0.7
    oscillators_4[4] = oscillators_4[4] - 1.4

    noise_rest = np.vstack(((
        np.array([random.gauss(0.0, 0.5) for i in range(200)]), np.array([random.gauss(0.0, 0.5) for i in range(200)]),
        np.array([random.gauss(0.0, 0.5) for i in range(200)]), np.array([random.gauss(0.0, 0.5) for i in range(200)]),
        np.array([random.gauss(0.0, 0.5) for i in range(200)]))))

    synthetic = np.concatenate(
        (noise_1, oscillators_1, noise_2, oscillators_2, noise_3, oscillators_3, noise_4, oscillators_4, noise_rest),
        axis=1)

    return synthetic


def fullSync_signal():
    coupled = dg.kuramoto_data(np.array([0, 0, 0, 0, 0]), [20, 20, 20, 20, 20], 0.6, 0, 0.01, 1240)
    coupled[1] = coupled[0] + 0.7
    coupled[2] = coupled[0] + 1.4
    coupled[3] = coupled[0] - 0.7
    coupled[4] = coupled[0] - 1.4

    return coupled

