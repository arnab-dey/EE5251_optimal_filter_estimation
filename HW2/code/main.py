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
dataFile = './inoutdata.txt'
##################################################################################
# Class definitions
##################################################################################
class LSQ:
    def __init__(self, dataFile, p=2, q=2):
        self.dataFile = dataFile
        # Check if data files are present in the location
        if not os.path.isfile(self.dataFile):
            print("Data file can't be located")
            exit(1)
        # Load training data and extract information
        self.data = np.loadtxt(self.dataFile)
        self.H = None
        self.x = None
        self.noisy_data = None
        self.p = p
        self.q = q

    ##################################################################################
    # This function add a column of synthetic noisy measurements
    ##################################################################################
    def add_noise(self, sigma):
        self.noisy_data = np.zeros((self.data.shape[0], 1))
        noise = sigma*np.random.normal(size=(self.data.shape[0],1))
        self.noisy_data[:, :] = self.data[:, 2].reshape((self.data.shape[0],1)) + noise
        # self.data = np.hstack((self.data,  noisy_y))

    ##################################################################################
    # This function returns input values and measurements
    ##################################################################################
    def get_data(self, idx):
        return self.data[idx, 1], self.noisy_data[idx, 0]

    ##################################################################################
    # This function updates the H matrix: Used in recursive least squares
    ##################################################################################
    def update_H_matrix(self, u, y):
        # Replacing new values with the oldest ones
        self.H[:, self.p-1] = -y
        self.H[:, self.p+self.q-1] = u
        # Doing a right shift to maintain the order
        self.H[:, 0:self.p] = np.roll(self.H[:, 0:self.p], 1)
        self.H[:, self.p:] = np.roll(self.H[:, self.p:], 1)

    ##################################################################################
    # This function performs recursive least squares
    ##################################################################################
    def perform_rls(self, sigma=0., is_noisy_data=False):
        ##################################################################################
        # initialization
        ##################################################################################
        if (False == is_noisy_data):
            self.add_noise(sigma)
        R = np.asarray([sigma**2]).reshape((1,1))
        self.H = np.zeros((1, self.p+self.q))
        self.x = np.zeros((self.p+self.q, 1))
        x_rec = np.zeros((self.data.shape[0], self.p+self.q))
        # assume that we do not know anything about inintial x
        # so let us inintialize P with a very high value
        init_factor = 1e+9
        P = init_factor*np.eye(self.p+self.q)
        prev_u = 2.
        prev_y = 2.
        for idx in range(self.data.shape[0]):
            ##################################################################################
            # get data first
            ##################################################################################
            u, y = self.get_data(idx)
            ##################################################################################
            # Update equations
            ##################################################################################
            self.update_H_matrix(prev_u, prev_y)
            if (idx >= int(self.p)):
                K = P @ self.H.T @ np.linalg.inv(self.H @ P @ self.H.T + R)
                # P = (np.eye(self.p+self.q) - K @ self.H) @ P
                P = (np.eye(self.p+self.q) - K @ self.H) @ P @ (np.eye(self.p+self.q) - K @ self.H).T + K @ R @ K.T
                self.x = self.x + K @ (y - self.H @ self.x)
            prev_u = u
            prev_y = y
            x_rec[idx, :] = self.x.T
        return self.x, x_rec

    ##################################################################################
    # This function performs ordinary least squares
    # it stacks all the time series data points in one sigle H matrix
    ##################################################################################
    def perform_ols(self, sigma, is_noisy_data=False, is_partial_data=False, time_idx=None):
        # self.H  = np.zeros((self.data.shape[0]-self.p, self.p+self.q))
        self.x = np.zeros((self.p + self.q, 1))
        if (False == is_noisy_data):
            self.add_noise(sigma)
        if (is_partial_data == False):
            self.H = np.zeros((self.data.shape[0] - self.p, self.p + self.q))
            for idx in range(self.data.shape[0]):
                if (idx >= self.p):
                    self.H[idx-self.p, 0] = -self.noisy_data[idx-1, 0]
                    self.H[idx-self.p, 1] = -self.noisy_data[idx-2, 0]
                    self.H[idx-self.p, 2] = self.data[idx-1, 1]
                    self.H[idx-self.p, 3] = self.data[idx-2, 1]
            self.x = np.linalg.inv(self.H.T @ self.H) @ self.H.T @ self.noisy_data[self.p:, :]
        else:
            if (time_idx is not None):
                self.H = np.zeros((time_idx.shape[0], self.p + self.q))
                h_idx = 0
                for idx in time_idx:
                    self.H[h_idx, 0] = -self.noisy_data[idx - 1, 0]
                    self.H[h_idx, 1] = -self.noisy_data[idx - 2, 0]
                    self.H[h_idx, 2] = self.data[idx - 1, 1]
                    self.H[h_idx, 3] = self.data[idx - 2, 1]
                    h_idx += 1
                y = self.noisy_data[time_idx, :]
                self.x = np.linalg.inv(self.H.T @ self.H) @ self.H.T @ self.noisy_data[time_idx, :]
            else:
                exit(1)
        return self.x

    ##################################################################################
    # This function predicts new data and returns root mean squared error
    ##################################################################################
    def predict(self, y, time_idx, is_training_data=False):
        # create H matrix first
        H = np.zeros((y.shape[0], self.p+self.q))
        h_idx = 0
        for idx in time_idx:
            H[h_idx, 0] = -self.noisy_data[idx-1, 0]
            H[h_idx, 1] = -self.noisy_data[idx-2, 0]
            H[h_idx, 2] = self.data[idx-1, 1]
            H[h_idx, 3] = self.data[idx-2, 1]
            h_idx += 1
        y_pred = H @ self.x
        # Calculate root mean squared error
        rmse = np.sqrt(np.mean((y_pred-y)**2))
        return rmse

##################################################################################
# CODE STARTS HERE
##################################################################################
np.random.seed(1)
ls = LSQ(dataFile, p=2, q=2)
# Part (b)
print("##### Runnning part (b) #####")
sigma = 0.
# RLS estimate
x, x_rec = ls.perform_rls(sigma)
print("RLS: a_1 = ", x[0])
print("RLS: a_2 = ", x[1])
print("RLS: b_1 = ", x[2])
print("RLS: b_2 = ", x[3])
# OLS estimate
x = ls.perform_ols(sigma, is_noisy_data=True)
print("Just for comparison with OLS...")
print("OLS: a_1 = ", x[0])
print("OLS: a_2 = ", x[1])
print("OLS: b_1 = ", x[2])
print("OLS: b_2 = ", x[3])
print('')

# Part (c), (d)
print("##### Runnning part (c), (d) #####")
sigma = 0.1
# RLS estimate
x, x_rec = ls.perform_rls(sigma)
print("RLS: a_1 = ", x[0])
print("RLS: a_2 = ", x[1])
print("RLS: b_1 = ", x[2])
print("RLS: b_2 = ", x[3])
# OLS estimate
x = ls.perform_ols(sigma, is_noisy_data=True)
print("Just for comparison with OLS...")
print("OLS: a_1 = ", x[0])
print("OLS: a_2 = ", x[1])
print("OLS: b_1 = ", x[2])
print("OLS: b_2 = ", x[3])
print('')

# Part (e)
print("##### Runnning part (e) #####")
sigma = 0.001
# RLS estimate
x, x_rec = ls.perform_rls(sigma)
print("RLS: a_1 = ", x[0])
print("RLS: a_2 = ", x[1])
print("RLS: b_1 = ", x[2])
print("RLS: b_2 = ", x[3])
# OLS estimate
x = ls.perform_ols(sigma, is_noisy_data=True)
print("Just for comparison with OLS...")
print("OLS: a_1 = ", x[0])
print("OLS: a_2 = ", x[1])
print("OLS: b_1 = ", x[2])
print("OLS: b_2 = ", x[3])
print('')

# Validating accuracy of the model
# DO 10 fold cross-validation for each sigma
print('##### Running 10-fold cross validation #######')
sigma_arr = np.asarray([0.1, 0.01, 0.001])
shift_idx = ls.p+1
N = ls.data.shape[0] - shift_idx
num_crossval = 10
blockSize = int((1/num_crossval)*N)
index = np.arange(N)
err_arr_val = np.zeros((len(sigma_arr), num_crossval))
for sigma_idx in range(sigma_arr.shape[0]):
    sigma = sigma_arr[sigma_idx]
    ls.add_noise(sigma)
    print('Running 10-fold cross validation for sigma = ', sigma)
    for fold in range(num_crossval):
        if (fold == num_crossval-1):
            v_idx = np.arange(fold * blockSize, N)
        else:
            v_idx = np.arange(fold*blockSize, (fold+1)*blockSize)
        t_idx = np.delete(index, v_idx)
        v_idx += shift_idx
        t_idx += shift_idx
        val_data = ls.noisy_data[v_idx, :]
        trn_data = ls.noisy_data[t_idx, :]
        x = ls.perform_ols(sigma, is_noisy_data=True, is_partial_data=True, time_idx=t_idx)
        err_val = ls.predict(val_data, v_idx)
        err_arr_val[sigma_idx, fold] = err_val
        print("sigma = ", sigma, "fold = ", fold, " validation RMSE = ", err_val)
        print('')

    print('##### Summary #####')
    print('sigma = ', sigma, 'average validation error = ', np.mean(err_arr_val[sigma_idx, :]))
    print('sigma = ', sigma, 'standard deviation of validation error = ', np.std(err_arr_val[sigma_idx, :]))
    print('')

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
    ax.errorbar(sigma_arr, np.mean(err_arr_val, axis=1), np.std(err_arr_val, axis=1).T, label='Validation error')

    ax.set_xlabel(r'sigma', fontsize=8)
    ax.set_ylabel(r'RMS error', fontsize=8)

    plt.legend()
    if (True == isPlotPdf):
        if not os.path.exists('./generatedPlots'):
            os.makedirs('generatedPlots')
        fig.savefig('./generatedPlots/val_error.pdf')
    else:
        plt.show()