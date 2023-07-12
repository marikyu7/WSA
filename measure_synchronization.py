import numpy as np
import pandas as pd
from multiSyncPy import synchrony_metrics as sm
import MultiSyncPyCI as msci
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (20 , 5)
import scipy

def signals_CI(data, fullSync_data, functions = [msci.symbolic_entropy, msci.sum_normalized_csd, msci.rho],
               functions_names = ['symbolic_entropy', 'sum_normalized_csd', 'rho'],
               window_sizes =  [12, 25, 37, 50, 62, 74, 87, 99, 112, 124]):
    function = []
    window_size = []
    time = []
    lower_bound = np.empty((0))
    upper_bound = np.empty((0))
    average = np.empty((0))
    lower_bound_fullSync = np.empty((0))
    upper_bound_fullSync = np.empty((0))
    average_fullSync = np.empty((0))

    for n in range(len(functions)):
        if functions[n] == msci.rho:
            synthetic = np.angle(scipy.signal.hilbert(data))
            fullSync = np.angle(scipy.signal.hilbert(fullSync_data))
        else:
            synthetic = data
            fullSync = fullSync_data
        for w in window_sizes:
            measure = sm.apply_windowed(
                synthetic,
                functions[n],
                window_length=w,
                step=1
            )
            l_b = measure[:, 1]
            u_b = measure[:, 2]
            av = measure[:, 0]
            w_s = [w] * measure.shape[0]
            f_n = [functions_names[n]] * measure.shape[0]
            t = []
            for m in range(measure.shape[0]):
                t.append(m)
            window_size = window_size + w_s
            lower_bound = np.concatenate([lower_bound, l_b])
            upper_bound = np.concatenate([upper_bound, u_b])
            average = np.concatenate([average, av])
            function = function + f_n
            time = time + t

            measure_fullSync = sm.apply_windowed(
                fullSync,
                functions[n],
                window_length=w,
                step=1
            )
            l_b_fullSync = measure_fullSync[:, 1]
            u_b_fullSync = measure_fullSync[:, 2]
            av_fullSync = measure_fullSync[:, 0]

            lower_bound_fullSync = np.concatenate([lower_bound_fullSync, l_b_fullSync])
            upper_bound_fullSync = np.concatenate([upper_bound_fullSync, u_b_fullSync])
            average_fullSync = np.concatenate([average_fullSync, av_fullSync])

    df = pd.DataFrame({'fnct': function,
                       'window_size': window_size,
                       'time': time,
                       'lower_bound': lower_bound,
                       'upper_bound': upper_bound,
                       'average': average,
                       'lower_bound_fullSync' : lower_bound_fullSync,
                       'upper_bound_fullSync': upper_bound_fullSync,
                       'average_fullSync': average_fullSync
                       })

    return df

def measure_distance_inclusion(data, fullSync_data,
                               functions = [msci.symbolic_entropy, msci.sum_normalized_csd, msci.rho],
                               window_sizes = [12, 25, 37, 50, 62, 74, 87, 99, 112, 124]):
    function = []
    window = []
    status = []
    sync_laps = []
    coup_per = []
    distance = []
    for n in range(len(functions)):
        if functions[n] == msci.rho:
            synthetic = np.angle(scipy.signal.hilbert(data))
            fullSync = np.angle(scipy.signal.hilbert(fullSync_data))
        else:
            synthetic = data
            fullSync = fullSync_data
        for w in window_sizes:
            m_synt = sm.apply_windowed(
                synthetic, functions[n], window_length=w, step=1)
            m_fullSync = sm.apply_windowed(
                fullSync, functions[n], window_length=w, step=1)
            av_sync = m_synt[:, 0]
            av_coup = m_fullSync[:, 0]
            if functions[n] == msci.symbolic_entropy:
                min_coup = m_fullSync[:, 2]
            else:
                min_coup = m_fullSync[:, 1]

            for m in range(m_synt.shape[0]):
                if functions[n] == msci.symbolic_entropy:
                    function.append('entropy')
                    coup_per.append(av_sync[m] < min_coup[m])
                else:
                    coup_per.append(av_sync[m] > min_coup[m])
                    if functions[n] == msci.rho:
                        function.append('rho')
                    elif functions[n] == msci.sum_normalized_csd:
                        function.append('csd')
                    else:
                        function.append('other')

                distance.append(abs(av_sync[m] - av_coup[m]))
                window.append(w)
                if (m) < 200:
                    status.append('noise')
                    sync_laps.append(0)
                elif (m) >= 200 and m < 220:
                    status.append('sync20')
                    sync_laps.append(20)
                elif (m) >= 220 and (m) < 420:
                    status.append('noise')
                    sync_laps.append(0)
                elif (m) >= 420 and (m) < 470:
                    status.append('sync50')
                    sync_laps.append(50)
                elif (m) >= 470 and (m) < 670:
                    status.append('noise')
                    sync_laps.append(0)
                elif (m) >= 670 and (m) < 740:
                    status.append('sync70')
                    sync_laps.append(70)
                elif (m) >= 740 and (m) < 940:
                    status.append('noise')
                    sync_laps.append(0)
                elif (m) >= 940 and (m) < 1040:
                    status.append('sync100')
                    sync_laps.append(100)
                elif (m) >= 1040:
                    status.append('noise')
                    sync_laps.append(0)

    df = {'function': function,
                       'window': window,
                       'status': status,
                       'sync_laps': sync_laps,
                       'coup_per': coup_per,
                       'distance': distance
                       }
    return df



