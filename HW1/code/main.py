##################################################################################
# IMPORTS
##################################################################################
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
################################################################################
# Settings for plotting
################################################################################
mpl.use('pdf')
fig_width  = 3.487
fig_height = fig_width / 1.618
rcParams = {
    'font.family': 'serif',
    'font.serif': 'Times',
    'text.usetex': True,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': [fig_width, fig_height]
   }
plt.rcParams.update(rcParams)
##################################################################################
# Dataset
##################################################################################
dataFile = './dataset.txt'
##################################################################################
# Class definitions
##################################################################################
class curveFit:
    def __init__(self, dataFile):
        self.dataFile = dataFile
        # Check if data files are present in the location
        if not os.path.isfile(self.dataFile):
            print("Data file can't be located")
            exit(1)
        # Load training data and extract information
        self.data = np.loadtxt(self.dataFile)
        self.H = None
    ##################################################################################
    # Function definitions
    ##################################################################################
    def _getMethod(self, method):
        if (method == 'linear'):
            return 0
        elif (method == 'quadratic'):
            return 1
        elif (method == 'cubic'):
            return 2
        elif (method == 'quartic'):
            return 3
        else:
            return -1

    ##################################################################################
    # This function fits the data and plots fitted curves
    ##################################################################################
    def fitData(self, method='linear', plotReqd=True, plotName='plot'):
        fitMethod = self._getMethod(method)
        if (fitMethod == -1):
            print('Invalid method')
            return
        N = self.data.shape[0]
        y = self.data[:, -1]
        y = np.reshape(y, (N, 1))
        t = self.data[:, 0]
        t = np.reshape(t, (N, 1))
        # Denoting starting year as 0: Coordinate transformation
        baseYear = t[0]
        t = t-baseYear
        # time axis for fitted curve plot
        numFitPoints = 500
        # For plotting purpose of fitted curve, extrapolating year axis from
        # 1945 to 1958 : THIS IS SOLELY FOR PLOTTING PURPOSE
        xFit = np.linspace(np.min(t)-1, np.max(t)+2, numFitPoints)
        xFit = np.reshape(xFit, (xFit.shape[0], 1))
        if (fitMethod == 0):
            self.H = np.hstack((np.ones((N, 1)), t))
            HFit = np.hstack((np.ones((numFitPoints, 1)), xFit))
            HPredict = np.array([1, t[-1]+1])
        elif (fitMethod == 1):
            self.H = np.hstack((np.ones((N,1)), t, t**2))
            HFit = np.hstack((np.ones((numFitPoints, 1)), xFit, xFit ** 2))
            HPredict = np.array([1, t[-1]+1, (t[-1]+1)**2])
        elif (fitMethod == 2):
            self.H = np.hstack((np.ones((N, 1)), t, t**2, t**3))
            HFit = np.hstack((np.ones((numFitPoints, 1)), xFit, xFit ** 2, xFit ** 3))
            HPredict = np.array([1, t[-1]+1, (t[-1]+1)**2, (t[-1]+1)**3])
        else:
            self.H = np.hstack((np.ones((N, 1)), t, t**2, t**3, t**4))
            HFit = np.hstack((np.ones((numFitPoints, 1)), xFit, xFit ** 2, xFit ** 3, xFit ** 4))
            HPredict = np.array([1, t[-1]+1, (t[-1]+1)**2, (t[-1]+1)**3, (t[-1]+1)**4])

        # Perform feature scaling: Required as we will be dealing with
        # values ranging from 0 to 10^4. Therefore, feature scaling will improve
        # the result for higher order polynomial. We are using simple min-max scaling.
        min = np.min(self.H[:, 1:], axis=0)
        max = np.max(self.H[:, 1:], axis=0)
        self.H[:, 1:] = (self.H[:, 1:]-min)/(max - min)
        HFit[:, 1:] = (HFit[:, 1:]-min)/(max - min)
        HPredict[1:] = (HPredict[1:]-min)/(max - min)
        H_L = np.linalg.inv(self.H.T @ self.H) @ self.H.T
        coef = H_L @ y
        yFit = HFit @ coef
        yHat = self.H @ coef
        rmsError = np.sqrt(np.mean((y-yHat)**2))
        print('Method = ', method, ', rms error = ', rmsError)
        print('Prediction of steel production in 1957 is ', (coef.T @ HPredict)[0][0])
        if (True == plotReqd):
            self._plotFit(y, yFit, t+baseYear, xFit+baseYear, plotName)


    def _plotFit(self, y, yFit, t, xFit, fileName):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle="-.", linewidth='0.1')

        # Plot datapoints
        ax.scatter(t, y, color='g')
        # Plot of fitted curve
        ax.plot(xFit, yFit, 'm', label='Fitted curve', linewidth=0.5)


        ax.set_xlabel(r'Year', fontsize=8)
        ax.set_ylabel(r'Production(million tons)', fontsize=8)
        plt.legend()
        plotName = './generatedPlots/' + fileName + '.pdf'
        fig.savefig(plotName)

##################################################################################
# CODE STARTS HERE
##################################################################################
methodArr = ['linear', 'quadratic', 'cubic', 'quartic']
cf = curveFit(dataFile)
for method in methodArr:
    cf.fitData(method=method, plotName='fit_'+method)