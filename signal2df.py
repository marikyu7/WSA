import random
import pandas as pd
import MultiSyncPyCI as msci
import measure_synchronization as sync
import plots
from matplotlib import pyplot as plt
import os

plt.rcParams["figure.figsize"] = (20 , 5)
random.seed(42)
dir_path = os.path.dirname(os.path.realpath('data'))

# generate data
clean_data = msci.signal()
noised_data = msci.signal_noised()
fullySync_data = msci.fullSync_signal()

# set parameters
functions = [msci.symbolic_entropy, msci.sum_normalized_csd, msci.rho]
functions_names = ['symbolic_entropy', 'sum_normalized_csd', 'rho']
window_sizes = [round(clean_data.shape[1]/100), round(clean_data.shape[1]/100*2), round(clean_data.shape[1]/100*3),
                     round(clean_data.shape[1]/100*4), round(clean_data.shape[1]/100*5), round(clean_data.shape[1]/100*6),
                     round(clean_data.shape[1]/100*7), round(clean_data.shape[1]/100*8), round(clean_data.shape[1]/100*9),
                     round(clean_data.shape[1]/100*10)]

# SYNTHETIC data
signals_CI_clean = sync.signals_CI(clean_data, fullySync_data)
signals_CI_clean.to_csv(dir_path + '/data/df_cleanSignal_CI.csv', index = False)

cleanSignal_df = sync.measure_distance_inclusion(clean_data, fullySync_data)
cleanSignal_df = pd.DataFrame(cleanSignal_df)
cleanSignal_df.to_csv(dir_path + '/data/cleanSignal_df.csv', index = False)

plots.plots(signals_CI_clean,
            titles = ['Symbolic Entropy - Synthetic signals', 'Rho - Synthetic signals',
                      'Sum Normalized CSD - Synthetic signals'],
            path = dir_path+'/figures/')

# NOISED data
signals_CI_noised = sync.signals_CI(noised_data, fullySync_data)
signals_CI_noised.to_csv(dir_path + '/data/df_noisedSignal_CI.csv', index = False)

noisedSignal_df = sync.measure_distance_inclusion(noised_data, fullySync_data)
noisedSignal_df = pd.DataFrame(noisedSignal_df)
noisedSignal_df.to_csv(dir_path + '/data/noisedSignal_df.csv', index = False)

plots.plots(signals_CI_noised,
            titles = ['Symbolic Entropy - Noised signals', 'Rho - Noised signals',
                      'Sum Normalized CSD - Noised signals'],
            path = dir_path+'/figures/')


# other PLOTS
plots.plot_fig_1(signals_CI_clean, clean_data)
plots.appendix_A(fullySync_data, clean_data, noised_data)