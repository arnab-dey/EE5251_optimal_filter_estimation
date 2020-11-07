##################################################################################
# IMPORTS
##################################################################################
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
################################################################################
# Variable declaration
################################################################################
isPlotReqd = True
isPlotPdf = True
################################################################################
# Settings for plot
################################################################################
if (True == isPlotReqd):
    if (True == isPlotPdf):
        mpl.use('pdf')
        fig_width  = 3.487
        fig_height = fig_width / 1.618
        rcParams = {
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            'text.usetex': True,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.labelsize': 8,
            'legend.fontsize': 8,
            'figure.figsize': [fig_width, fig_height]
           }
        plt.rcParams.update(rcParams)
##################################################################################
# PARAMETERS
##################################################################################
T = 1.
Q = 1.
t_0 = 0
t_N = 50
t = np.linspace(t_0, t_N, num=t_N+1)
N = len(t)
num_run = 100
x_0 = 0.
np.random.seed(1)
##################################################################################
# PROCESS NOISE
##################################################################################
w_k = np.sqrt(Q) * np.random.normal(0, 1, (N-1, num_run)) # We dont have any process noise at t=0
w_k = np.vstack((np.zeros((1, num_run)), w_k))
##################################################################################
# SYSTEM STATES
##################################################################################
x_k = np.cumsum(w_k, axis=0)
#######################################################################
# Plot of error curves
#######################################################################
if (True == isPlotReqd):
    ###########################################################################
    # Configure axis and grid
    ###########################################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5')
    ax.grid(which='minor', linestyle="-.", linewidth='0.5')
    for run in range(num_run):
        ax.plot(t, x_k[:, run])

    ax.set_xlabel(r'$t_k$', fontsize=8)
    ax.set_ylabel(r'$x_k$', fontsize=8)

    # plt.legend()
    if (True == isPlotPdf):
        if not os.path.exists('./generatedPlots'):
            os.makedirs('generatedPlots')
        fig.savefig('./generatedPlots/q2_d.pdf')
    else:
        plt.show()
##################################################################################
# ENSEMBLE standard deviation
##################################################################################
print('Ensemble standard deviation at t=5 is ', np.std(x_k[5, :]))
print('Ensemble standard deviation at t=25 is ', np.std(x_k[25, :]))
print('Ensemble standard deviation at t=50 is ', np.std(x_k[50, :]))
##################################################################################
# (e) COVARIANCE ANALYSIS
##################################################################################
P = np.ones((N-1, 1))
P = np.vstack((np.zeros((1,1)), P))
P_k = np.cumsum(P)
sigma_k = np.sqrt(P_k)
#######################################################################
# Plot of error curves
#######################################################################
if (True == isPlotReqd):
    ###########################################################################
    # Configure axis and grid
    ###########################################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5')
    ax.grid(which='minor', linestyle="-.", linewidth='0.5')
    for run in range(num_run):
        ax.plot(t, x_k[:, run])
    ax.plot(t, sigma_k, color='k', label='$\pm \sigma_k$')
    ax.plot(t, -sigma_k, color='k')
    ax.set_xlabel(r'$t_k$', fontsize=8)
    ax.set_ylabel(r'$x_k$', fontsize=8)

    plt.legend()
    if (True == isPlotPdf):
        if not os.path.exists('./generatedPlots'):
            os.makedirs('generatedPlots')
        fig.savefig('./generatedPlots/q2_e.pdf')
    else:
        plt.show()

print('Covariance analysis: standard deviation at t=5 is ', sigma_k[5])
print('Covariance analysis: standard deviation at t=10 is ', sigma_k[10])