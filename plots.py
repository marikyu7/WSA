from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (20 , 5)


def plots(df, enthropy = 'symbolic_entropy', rho = 'rho', csd = 'sum_normalized_csd',
          time_col = 'time', lb_col = 'lower_bound_fullSync', ub_col = 'lower_bound_fullSync', sign_col = 'average',
          titles = ['Symbolic Entropy - Noised signals', 'Rho - Noised signals', 'Sum Normalized CSD - Noised signals'],
          window_sizes = [12, 25, 37, 50, 62, 74, 87, 99, 112, 124],
          show = False, save = True, path = '/figures/'):
    enthropy = df.copy()[df['fnct'] == enthropy]
    rho = df.copy()[df['fnct'] == rho]
    csd = df.copy()[df['fnct'] == csd]
    dfs = [enthropy, rho, csd]

    def update_axis(ax, df, window):
        clr = plt.cm.Purples(0.9)
        t = 'window size: ' + str(window)
        ax.set_title(t, fontsize=7, fontweight='bold')
        x = df[time_col]
        y_l_s = df[lb_col]
        y_m = df[sign_col]
        y_u_s = df[ub_col]
        ax.plot(x, y_m, label='average', color=clr)
        ax.fill_between(x, y_l_s, y_u_s, alpha=0.3, edgecolor=clr, facecolor=clr)
        ax.set_ylabel('sync', fontsize='medium')
        ax.set_xlabel('t', fontsize='medium')
        ax.tick_params(axis='both', labelsize='small')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    for n in range(len(titles)):
        fig, axs = plt.subplots(3, 3,
                                sharex=True, sharey=True,
                                figsize=(20, 14))  # ,
        # facecolor = plt.cm.Blues(.2))
        fig.tight_layout(pad=5.0)
        title = titles[n]
        fig.suptitle(title, fontsize='xx-large', fontweight='bold')
        for i, ax in enumerate(axs.flatten()):
            if i > 8:
                break
            else:
                df_c = dfs[n][dfs[n]['window_size'] == window_sizes[i]]
                update_axis(ax, df_c, window_sizes[i])
        if show == True:
            plt.show()
        if save == True:
            plt.savefig(path + titles[n] + '.pdf')

def plot_fig_1(df, clean_sign, function_col = 'fnct', function_name = 'symbolic_entropy', window_size = 12,
               titles = ["a) Synthetic signals set", 'b) Symbolic Entropy - Synthetic signals'],
               time_col = 'time', lb_col = 'lower_bound_fullSync', ub_col = 'upper_bound_fullSync', sign_col = 'average'):
    function_df = df.copy()[df[function_col]==function_name]
    def update_axis(ax, df):
        clr = plt.cm.Purples(0.9)
        ax.set_title(titles[1])
        x = df[time_col]
        y_l_s = df[lb_col]
        y_m = df[sign_col]
        y_u_s = df[ub_col]
        ax.plot(x, y_m, label='average', color=clr)
        ax.fill_between(x, y_l_s, y_u_s, alpha=0.3, edgecolor=clr, facecolor=clr)
        ax.set_ylabel('sync', fontsize='medium')
        ax.set_xlabel('t', fontsize='medium')
        ax.tick_params(axis='both', labelsize='small')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    figure, axis = plt.subplots(2, 1)
    axis[0].set_title(titles[0])
    axis[0].plot(clean_sign.T, alpha = 0.7)
    axis[0].tick_params(axis='both', labelsize='small')
    axis[0].spines['right'].set_visible(False)
    axis[0].spines['top'].set_visible(False)

    df_c = function_df[function_df['window_size']== window_size]
    update_axis(axis[1], df_c)

    plt.show()


def appendix_A(fullSync_signal, clean_sign, noised_sign):
    figure, axis = plt.subplots(3, 1)

    coupled_data = fullSync_signal
    axis[0].set_title("a) Synchronized signals set")
    axis[0].plot(coupled_data.T, alpha = 0.7)
    axis[0].tick_params(axis='both', labelsize='small')
    axis[0].spines['right'].set_visible(False)
    axis[0].spines['top'].set_visible(False)

    synthetic_data = clean_sign
    axis[1].set_title("b) Synthetic signals set")
    axis[1].plot(synthetic_data.T, alpha = 0.7)
    axis[1].tick_params(axis='both', labelsize='small')
    axis[1].spines['right'].set_visible(False)
    axis[1].spines['top'].set_visible(False)

    synthetic_noised_data = noised_sign
    axis[2].set_title("c) Synthetic noised signals set")
    axis[2].plot(synthetic_noised_data.T, alpha = 0.7)
    axis[2].tick_params(axis='both', labelsize='small')
    axis[2].spines['right'].set_visible(False)
    axis[2].spines['top'].set_visible(False)
