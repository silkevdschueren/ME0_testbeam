import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import uproot
import awkward as ak
from scipy import stats
import scipy.odr as odr
from matplotlib.ticker import FormatStrFormatter
from itertools import compress
import math
import scipy as sc
import pandas as pd
from itertools import accumulate



def read_runfile(runnumber, downloaded=False, printbranches=False):
    """
    Read the content of the given runfile and print the branches.
    
    :param runnumber: Number of the file you want to read as string of three digits, ex 00000010 has to be given as '010'.
    :param downloaded: Optional parameter to indicate if file was downloaded in local eos. Default is False.
    :param printbranches: Optional parameter to indicate if branches have to be printed. Default is False.
    :return: The branches of the file.
    """
    
    if not downloaded:
        file = uproot.open(f'/eos/cms/store/group/upgrade/GEM/TestBeams/July2022/tracks/00000{runnumber}.root')
    else:
        file = uproot.open(f'00000{runnumber}.root')
        
    tree = file['trackTree']
    branches = tree.arrays()#entry_stop=100000
    
    if printbranches:
        print(tree.keys())
    
    return branches



def calc_residuals(branches):
    """
    Apply a chi2 cut to exclude events with chi2 equals to zero, determine propagated and reconstructed hits 
    and filter on valid events and calculate the residuals. 
    
    :param branches: Branches of the event you want to reduce.
    :return: prop_x, prop_y, rec_x, rec_y, residuals_x, residuals_y: Propagated and reconstructed hit positions and residuals, 
                                                                     all as awkward arrays; 
             valid_props, valid_recs: Mask on events with propagated and reconstructed hits, as awkward array mask for events;
             chi2_mask: Mask on events with chi2 not equal to zero, as awkward array mask for events.
    """
    
    # Filter chi2>0, the tracks with chi2=0 are those in which the reconstruction failed. 
    # This mask will not be used for efficiency calculations.
    chi2_x_cut = 0
    chi2_y_cut = 0
    chi2_x_mask = branches['trackChi2X'] > chi2_x_cut
    chi2_y_mask = branches['trackChi2Y'] > chi2_y_cut
    chi2_mask = chi2_x_mask & chi2_y_mask

    # X and y coordinates of the propagated hit. Propagated hits are constructed global.
    prop_x = branches['prophitGlobalX'][chi2_mask]
    prop_y = branches['prophitGlobalY'][chi2_mask]

    # X and y coordinates of all reconstructed hits. Reconstructed hits are local.
    rec_x = branches['rechitLocalX'][chi2_mask]
    rec_y = branches['rechitLocalY'][chi2_mask]

    # Filter on valid events, so events with propagated hits. Also determine filter for events with reconstructed hits. 
    # I will use only the events with propagated hits for most of the analysis, 
    # for the efficiency calculations I will not mask these.
    valid_props = ~ak.is_none(ak.firsts(prop_x))
    valid_recs = ~ak.is_none(ak.firsts(rec_x))

    rec_x = rec_x[valid_props]
    rec_y = rec_y[valid_props]
    prop_x = prop_x[valid_props]
    prop_y = prop_y[valid_props]

    # Detector was mirrored compared to trackers.
    prop_x = -prop_x
    
    # Determine residuals in x and y.
    residuals_x = rec_x - ak.broadcast_arrays(ak.flatten(prop_x), rec_x)[0]
    residuals_y = rec_y - ak.broadcast_arrays(ak.flatten(prop_x), rec_x)[0]
    
    
    return prop_x, prop_y, rec_x, rec_y, residuals_x, residuals_y, valid_props, valid_recs, chi2_mask



def gaussian_linear(x, a, x0, sigma, b, c):
    """
    Model of gaussian peak with linear background.
    
    :param x: Given x position to calculate value of the gaussian with linear background.
    :param a: Amplitude of the gaussian.
    :param x0: Mean value of the gaussian.
    :param sigma: Standard deviation of the gaussian
    :param b: Slope of the linear background.
    :param c: Offset of the linear background.
    :return: Value of the gaussian with linear background in the given point.
    """
    
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + b*x + c



def gaussian_exponential(x, a, x0, sigma, b, c):
    """
    Model of gaussian peak with exponential background.
    
    :param x: Given x position to calculate value of the gaussian with exponential background.
    :param a: Amplitude of the gaussian.
    :param x0: Mean value of the gaussian.
    :param sigma: Standard deviation of the gaussian
    :param b: Decay of the exponential background.
    :param c: Aplitude of the exponential background.
    :return: Value of the gaussian with exponential background in the given point.
    """
    
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c*np.exp(-b*x)



def fit_gaussian(runnumber, branches, residuals_x, valid_props, chi2_mask, show=False):
    """
    Fit a gaussian peak with exponential and linear background to the residuals.
    
    :param runnumber: Number of the run to fit.
    :param branches: Branches of the event.
    :param residuals_x: Residuals in the x direction. 
    :param valid_props: Mask to select events with valid propagated hits.
    :param chi2_mask: Mask to select events with nonzero chi2.
    :param show: Optional parameter to indicate if you want to show the figures of the plot. Default False.
    :return: xres_mean, xres_sigma: Mean and sigma from gaussian fit with exponential background, 
             tuples with data for both etapartitions;
             xres_mean_std, xres_sigma_std: Standard deviations of the fit parameters, tuples for both etapartitions.
    """
    
    # arrays to the mean and standard deviation of the gaussian for both etapartitions (1 and 2), 
    # with index the number of the etapartition - 1.
    xres_mean = np.zeros(2)
    xres_sigma = np.zeros(2)
    xres_mean_std = np.zeros(2)
    xres_sigma_std = np.zeros(2)

    # Apply same masks to the reconstructed eta as applied to the residuals, to keep the same events in the same order.
    rechit_eta = branches['rechitEta'][chi2_mask][valid_props]

    # There are two etapartitions that are illuminated, so create two figures for fits.
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    for eta in (1, 2):
        # Mask on residuals for which reconstructed hit is in the given etapartition.
        eta_mask = rechit_eta == eta

        # Bins for residual distribution.
        minbin = -100
        maxbin = 20
        binsize = 1
        bins = np.arange(minbin-binsize/2, maxbin+binsize/2+1, binsize)

        # Plot the residual distribution and save the data from the histogram.
        fitdata, fitbins, _ = ax[eta-1].hist(ak.flatten(residuals_x[eta_mask]), bins=bins)
        bincenters = fitbins[:-1] + binsize/2

        # Fit gaussian distribution with either linear or exponential background to the data. 
        # The guess for the linear distribution is made for one run 
        # and expected to converge for other runs as well. 
        # The guess for the exponential distribution is given by the parameters of the linear distribution, 
        # since at least the gaussian part and the amplitude of the background are the same.
        parameters_linear, covariance_linear = curve_fit(gaussian_linear, bincenters, fitdata, p0=[450, -37, 0.5, 0, 40], 
                                                         bounds = ([0, -45, 0, -20, 0], [100000, -30, 5, 500, 100]))
        parameters_exponential, covariance_exponential = curve_fit(gaussian_exponential, bincenters, fitdata,
                                                                   p0=parameters_linear)

        # Plot the fits on the histograms.
        ax[eta-1].plot(np.arange(minbin, maxbin, 0.001), gaussian_exponential(np.arange(minbin, maxbin, 0.001), 
                                                                              *parameters_exponential), 
                       label=f"gauss with mean {parameters_exponential[1]:.3f} and standard deviation {parameters_exponential[2]:.3f}, exponential background", 
                       lw=2, ls="-", color='red')

        # Set the figures.
        ax[eta-1].set(xlabel='Residuals in x direction', ylabel='Events', 
                      title=f"{runnumber} Distribution of residuals in x, eta = {eta}", ylim=(0, 1.2*np.max(fitdata)))
        ax[eta-1].legend(bbox_to_anchor=[1, 1], loc='upper right')

        # Save mean and standard deviation.
        xres_mean[eta-1] = parameters_exponential[1]
        xres_sigma[eta-1] = abs(parameters_exponential[2])
        xres_mean_std[eta-1] = np.sqrt(np.diag(covariance_exponential))[1]
        xres_sigma_std[eta-1] = np.sqrt(np.diag(covariance_exponential))[2]
    
    # Show the figures only if explicitly asked.
    if show:
        plt.show()
    plt.close()
    
    
    return xres_mean, xres_sigma, xres_mean_std, xres_sigma_std



def calc_efficiency(runnumber, sigmanum, branches, residuals_x, valid_recs, valid_props, chi2_mask):
    """
    Calculate the efficiency for the ME0 for the given event using the tracks, for both etapartitions and for the whole detector.
    
    :param runnumber: Number of the run to fit.
    :param sigmanum: Allowed number of sigmas from the mean value in the residual distribution for a proper reconstructed track.
    :param branches: Branches of the event.
    :param residuals_x: Residuals in the x direction. 
    :param valid_recs: Mask to select events with valid reconstructed hits.
    :param valid_props: Mask to select events with valid propagated hits.
    :param chi2_mask: Mask to select events with nonzero chi2.
    :return: efficiencies: List with efficiencies, for etapartition one, etapartition twom whole detector;
             erros: List with errors on the efficiency from binomial distribution, 
                    for etapartition one, etapartition twom whole detector.
    """
    
    # Determine parameters of the fit of the residual distribution.
    xres_mean, xres_sigma, xres_mean_std, xres_sigma_std = fit_gaussian(runnumber, branches, residuals_x, valid_props, chi2_mask)
    
    # Apply same masks to the reconstructed eta as applied to the residuals, to keep the same events in the same order.
    prophit_eta = branches['prophitEta'][chi2_mask][valid_props]
    
    # Lists for etapartition one, etapartition two, whole detector.
    efficiencies = np.zeros(3)  # Efficiencies.
    errors = np.zeros(3)  # Errors on the efficiencies.
    k = np.zeros(3)  # Number of events that survives the cut.
    N = np.zeros(3)  # Total number of events.
    
    # EFFICIENCY FOR ETAPARTITIONS.
    for eta in (1, 2):
        # The label of eta for propagated hits is given by the one of the reconstructed hits plus seven.
        prophit_eta_mask = prophit_eta == eta+7

        # The total number of events is given by the number of events with propagated hit in this etapartition.
        N[eta-1] = np.sum(prophit_eta_mask)
        N[2] += N[eta-1]

        # Mask on proper reconstructed tracks for this eta. This mask is on residuals or rechits, not on events.
        properrec_mask = abs(residuals_x[valid_recs & ak.flatten(prophit_eta_mask)] - xres_mean[eta-1]) < sigmanum * xres_sigma[eta-1]
        
        # Determine number of events with proper reconstructed tracks, so see if there is a reconstructed hit 
        # which is recognised as muon in the event.
        k[eta-1] = np.sum(np.any(properrec_mask, axis=1))
        k[2] += k[eta-1]

        # Calculate the efficiency.
        efficiencies[eta-1] = k[eta-1] / N[eta-1]
        
        # Calculate the errors.
        errors[eta-1] = (1/N[eta-1])*np.sqrt(k[eta-1]*(1 - k[eta-1]/N[eta-1]))
        
    # TOTAL EFFICIENCY OF THE DETECTOR
    efficiencies[2] = k[2] / N[2]

    # Calculate the errors.
    errors[2] = (1/N[2])*np.sqrt(k[2]*(1 - k[2]/N[2]))

    return efficiencies, errors



def reconstruct_time(branches):
    """
    Reconstruct the time from bunchcounter and eventcounter.
    
    :param branches: The branches of the event.
    :return: Timestamps
    """
    
    # Bunch crossing and orbit frequency from the LHC.
    bunchcrossing_frequency = 40.079  # MHz
    bunchcrossing_time = 1/bunchcrossing_frequency * 10**3  # ns
    orbit_frequenty = 11.245  # kHz
    orbit_time = 1/orbit_frequenty * 10**6  # ns

    return (branches['orbitNumber'] * orbit_time + branches['bunchCounter'] * bunchcrossing_time)



def zeros_count(a):
    """
    Determine the number of zeros between two nonzero values, including beginning and end as nonzero.
    
    :param a: Numpy array.
    :return: Numpy array of number of zeros betweem two nonzero values.
    """
    
    idx = np.flatnonzero(a)
    intervals = np.concatenate((idx[1:] - idx[:-1], [len(a)-idx[-1]]))
    return intervals if idx[0] == 0 else np.concatenate(([idx[0]], intervals))



def time_from_spillstart(branches):
    """
    Reconstruct the time from the beginning of the spill for each event. Take into account that the orbit timer can be reset 
    during a spill (time has to continue increasing) or interspill (new spill after reset).
    
    :param branches: The branches of the event.
    :return: array of times since beginning of last spill for all events.
    """
    
    # Bunch crossing and orbit frequency from the LHC.
    bunchcrossing_frequency = 40.079  # MHz
    bunchcrossing_time = 1/bunchcrossing_frequency * 10**3  # ns
    orbit_frequenty = 11.245  # kHz
    orbit_time = 1/orbit_frequenty * 10**6  # ns
    
    # Some data concerning time.
    timestamps = reconstruct_time(branches)
    orbitnumber = branches['orbitNumber']
    bunchcounter = branches['bunchCounter']
    eventcounter = branches['eventCounter']
    
    # Number of orbitcounter resets during the current spill.
    resetcounter = 0
    # Value at which the orbitcounter will reset, here 16 bits.
    maxorbitvalue = 2**16
    
    # If orbit is reset during spill, orbitcounter decreases by value bigger than orbit_jump.
    orbit_jump = maxorbitvalue - 500
    # New spill without orbit reset.
    spill_jump = 1000
    
    # Orbit counter reset when orbit counter decreases, both during spill and interspill.
    orbit_reset = np.concatenate(([True],(orbitnumber[:-1] > orbitnumber[1:])))
    # Beginning of a new spill if either the orbitnumber decreases with a value lower than the maximal value, 
    # or the orbitcounter makes a jump.
    new_spill = np.concatenate(([True], (((orbitnumber[:-1] > orbitnumber[1:]) & (orbitnumber[:-1] - orbitnumber[1:] <= orbit_jump)) | (orbitnumber[1:] - orbitnumber[:-1] > spill_jump))))

    # Always increasing times, orbit resets taken into account.
    orbit_reset_accum = np.array(list(accumulate([int(item) for item in orbit_reset])))
    times = timestamps + maxorbitvalue*orbit_reset_accum*orbit_time

    # Separate the different spills.
    # Spillbegins [t1, t2, t3]
    begintimes = times[new_spill]
    # [t1 t1 t1 t1 t2 t2 t2 t2 t3 t3 t3 t3 t3]
    multiples = zeros_count(np.array(new_spill))
    begins = [t for t, num in zip(begintimes, multiples) for _ in range(num)]

    spilltimes = times - begins
            
    return spilltimes



def plot_meaneff_spilltime(branches, sigmanum = 3, N = 100, xmax = 2000, makefit=False):
    """
    Plot the mean value of the coarse efficiency over different spills as a function of the time since the beginning of a spill. 
    
    :param branches: Branches of the event.
    :param sigmanum: Allowed number of sigmas from the mean value in the residual distribution 
                     for a proper reconstructed track.
    :param N: Number of successive events taken together in one bin to calculate efficiency.
    :param xmax: Maximal x value of the plot.
    :param fit: Optional parameter to make exponentially decreasing fit to the data.
    :return: 
    """

    # Make figure to plot the data.
    fig = plt.figure(figsize=(25, 7))
    ax = fig.add_subplot(111)
    
    # Empty lists to save data to plot.
    efficiencies = []
    timestamps = []
    xerrors = []
    yerrors = []

    # Time since beginning of spill.
    spilltime = time_from_spillstart(branches)
    
    # Hits to calculate coarse efficiency.
    rechits = branches['rechitLocalX']
    
    # Order all events from all spills chronologically since the beginning of the spill.
    sort = np.argsort(spilltime)
    spilltime = spilltime[sort]

    newrechits = ak.ArrayBuilder()
    for index in sort:
        newrechits.append(rechits[index])
    rechits = newrechits
    
    # Loop over all events in chronological order and determine the efficiency for bins of events.
    for index in range(0, len(spilltime)-N, N):

        # Calculate the coarse efficiency.
        k = np.sum(~ak.is_none(ak.firsts(rechits[index:index+N])))
        efficiency = k / N
        efficiencies.append(efficiency)

        # Keep mean values of timesteps.
        timestamps.append((spilltime[index+N] + spilltime[index])/2)
        
        # Errors in x direction.
        xerrors.append((spilltime[index+N] - spilltime[index])/np.sqrt(12))

        # Errors in y direction (https://lss.fnal.gov/archive/test-tm/2000/fermilab-tm-2286-cd.pdf).
        yerrors.append((1/N)*np.sqrt(k*(1 - k/N)))
        
    ax.errorbar(timestamps, efficiencies, xerr=xerrors, yerr=yerrors, ls='', capsize=3)
    ax.set(xlabel="Time (ns)", ylabel="Efficiency (%)", title=f"Efficiency of the ME0 at {sigmanum} sigma", ylim=(0.8, 1))
    
    if makefit:
        myoutput = fit_exponential(timestamps, efficiencies, xerrors, yerrors, 
                                   guess=[np.max(efficiencies), 0, 0])
        ax.plot(np.linspace(0, np.max(timestamps), 100), exponential(myoutput.beta, np.linspace(0, np.max(timestamps), 100)),
               label=f'{myoutput.beta[0]:.2f} * exp(-{myoutput.beta[1]:.2e})t + {myoutput.beta[2]:.2e}')
    
    plt.legend()
    plt.show()
    
    return



def read_runinfo(filename):
    """
    Read all needed information about all testbeam runs from the given csv file, remove bad runs 
    and adjust runnumbers so that they are strings of length three.
    
    :param filename: Name of the excel file.
    :return: Pandas dataframe with information from the file.
    '"""
    
    file = pd.read_csv("GEM GIF++ testbeam 2022 - Runs.csv")
    file = file.drop(file[(file['GOOD run?'] == 'no')].index)
    file = file.drop(0, "index")
    
    # Adjust strings so that all of them have length three, in order to use them as filenames.
    for index, _ in file.iterrows():
        if file["Run"][index] == file["Run"][index]:  # NaN is not equal to itself.
            file.loc[index,"Run"] = str(int(float(file["Run"][index]))).rjust(3, '0')
    
    return file



def remove_run(file, runnumber):
    """
    Remove all data from given pandas dataframe of a given run because there is something wrong with it.
    
    :param file: Pandas dataframe from which you want to remove the information about the run.
    :param runnumber: Number of the run to remove, as string of length three.
    :return: /
    """
    
    return file.drop(file[(file['Run'] == runnumber)].index)



def add_efficiency(file, sigmanum=1.5):
    """
    Calculate the efficiency of all runs taken.
    
    :param file: Pandas dataframe with information about the runs.
    :param sigmanum: Optional parameter giving which tracks are considered to be proper reconstructed.
    :return: file: Pandas dataframe with information about the runs with efficiency and errors added.
    """
    
    efficiencies = []
    errors = []

    # Remove run 381 since this file is broken and does not fall in exceptions.
    file = remove_run(file, '381')

    for index, run in file.iterrows():
        try:
            # Calculate needed information.
            runnumber = run["Run"]
            branches = read_runfile(runnumber)
            prop_x, prop_y, rec_x, rec_y, residuals_x, residuals_y, valid_props, valid_recs, chi2_mask = calc_residuals(branches)
            efficiency, error = calc_efficiency(runnumber, sigmanum, branches, residuals_x, valid_recs, valid_props, chi2_mask)

            # Save total efficiency for all runs.
            efficiencies.append(efficiency[2])
            errors.append(error[2])

        except FileNotFoundError:
            file = remove_run(file, runnumber)
            print(f"File for run {runnumber} not found.")

        except RuntimeError:
            file = remove_run(file, runnumber)
            print(f"Optimal parameters for run {runnumber} not found")

        #except NameError:
            #remove_run(runnumbers.index(runnumber))
            #print(f"No keys for run {runnumber}.")      
            
    file = file.assign(calculated_efficiencies = efficiencies)
    file = file.assign(efficiency_errors = errors)
    
    return file



def read_rateatt(filename):
    """
    Read txt file containing the attenuation, rate and rate errors.
    
    :param filename: Name of the file containing the attenuation, rate and rate errors.
    :return: Dictionary with keys the attenuations and items the rates and the errors.
    """
    
    ratefile = open(filename, 'r')
    ratedata = ratefile.read().split('\n')
    
    attenuation_rate = dict()
    
    for line in ratedata[1:-1]:
        attenuation, rate, rate_errors = line.split(',')
        attenuation_rate[attenuation] = (float(rate), float(rate_errors))
    
    return attenuation_rate



def add_attenuation(file, att_rate_file):
    """
    Add columns with the rate of the source and error on the rate of the source to the pandas dataframe.
    
    :param file: Pandas dataframe with information on the runs.
    :param filename: Name of the file containing the attenuation, rate and rate errors.
    :return: Pandas dataframe where the rate and rate errors are added.
    """
    
    attenuation = file["attenuation"]
    att_rate = read_rateatt(att_rate_file)
    rates = [att_rate[att][0] if att in att_rate.keys() else 0 for att in attenuation]
    rate_errors = [att_rate[att][1] if att in att_rate.keys() else 0 for att in attenuation]
    
    file = file.assign(source_rate = rates)
    file = file.assign(source_rate_error = rate_errors)
    
    return file



def exponential(params, x):
    """
    Exponential function.
    
    :param params: List [a, b, c] with amplitude, decay time and offset of the exponential.
    :param x: Point in which to determine the value of the exponential.
    :return: Value of the exponential.
    """
    
    return params[0]*np.exp(-x*params[1]) + params[2]



def fit_exponential(xvalues, yvalues, xerrors=None, yerrors=None, guess=None):
    """
    Fit an exponential function to the given data.
    
    :param xvalues: List of x values of the data.
    :param yvalues: List of y values of the data.
    :param xerrors: List of errors on datapoints in x direction. Optional.
    :param yerrors: List of errors on datapoints in y direction. Optional.
    :param guess: Initial guess for the parameters of the exponential function. 
    :return: scipy.odr.Output class, with estimated parameters in '.beta'.
    """
    
    # Create a Model.
    exp_model = odr.Model(exponential)

    # Create a Data or RealData instance.
    mydata = odr.RealData(xvalues, yvalues, sx=xerrors, sy=yerrors)

    # Instantiate ODR with your data, model and initial parameter estimate.
    myodr = odr.ODR(mydata, exp_model, beta0=guess)

    # Run the fit.
    myoutput = myodr.run()

    return myoutput



def mean_chi2(branches):
    """
    Determine the mean chi2 value of the given event, taking into account a cut on the chi2 distribution.
    
    :param branches: Branches of the event.
    :return: Mean value of the chi2 of the tracks in the event.
    """

    chi2_x_cut = 5
    chi2_x_mask1 = branches['trackChi2X'] > 0
    chi2_x_mask2 = branches['trackChi2X'] < chi2_x_cut
    chi2_y_cut = 5
    chi2_y_mask1 = branches['trackChi2Y'] > 0
    chi2_y_mask2 = branches['trackChi2Y'] < chi2_y_cut

    chi2_mask = chi2_x_mask1 & chi2_y_mask1 & chi2_x_mask2 & chi2_y_mask2

    return np.mean(np.sqrt(branches['trackChi2X'][chi2_mask]**2 + branches['trackChi2Y'][chi2_mask])**2)



def add_chi2(file):
    """
    Calculate the mean chi2 of all runs taken.
    
    :param file: Pandas dataframe with information about the runs.
    :return: Pandas dataframe with information about the runs with mean chi2 added.
    """
    
    # Remove run 381 since this file is broken and does not fall in exceptions.
    file = remove_run(file, '381')
    chi2s = []

    for index, run in file.iterrows():
        try:
            # Calculate needed information.
            runnumber = run["Run"]
            branches = read_runfile(runnumber)
            chi2 = mean_chi2(branches)
            chi2s.append(chi2)

        except FileNotFoundError:
            file = remove_run(file, runnumber)
            #print(f"File for run {runnumber} not found.")

        except RuntimeError:
            #file = remove_run(file, runnumber)
            chi2s.append("not found")
            #print(f"Optimal parameters for run {runnumber} not found")

        #except NameError:
            #remove_run(runnumbers.index(runnumber))
            #print(f"No keys for run {runnumber}.")

    file = file.assign(mean_chi2 = chi2s)

    return file



def add_rotation(file, show=False):
    """
    Add the rotation of all runs taken to the pandas dataframe
    
    :param file: Pandas dataframe with information about the runs.
    :param show: Parameter indicating if all plots have to be shown. Default False.
    :return: Pandas dataframe with information about the runs with rotation in radians added.
    """
    
    # Remove run 381 since this file is broken and does not fall in exceptions.
    file = remove_run(file, '381')
    rotations = []
    yoffsets = []

    for index, run in file.iterrows():
        try:
            # Calculate needed information.
            runnumber = run["Run"]
            branches = read_runfile(runnumber)
            rotation, yoffset = calc_rotation_yoffset(branches, show)
            rotations.append(rotation)
            yoffsets.append(yoffset)

        except FileNotFoundError:
            file = remove_run(file, runnumber)
            #print(f"File for run {runnumber} not found.")

        except RuntimeError:
            #file = remove_run(file, runnumber)
            rotations.append("not found")
            yoffsets.append("not found")
            #print(f"Optimal parameters for run {runnumber} not found")

        #except NameError:
            #remove_run(runnumbers.index(runnumber))
            #print(f"No keys for run {runnumber}.")

    file = file.assign(rotation = rotations)
    file = file.assign(yoffset = yoffsets)

    return file



def linear(B, x):
    """
    Linear function y = m*x + b
    :param B: Vector of the parameters.
    :param x: Array of the current x values. x is in the same format as the x passed to Data or RealData.
    :return: Array in the same format as y passed to Data or RealData.
    """

    return B[0]*x + B[1]



def fit_line(x, y, xerrors, yerrors):
    """
    Fit a line through the given points, taking into account both x and y errors.
    
    :param x: X positions of the points as a list.
    :param y: Y positions of the points as a list.
    :param xerrors: X errors of the points as a list.
    :param yerrors: Y errors of the points as a list.
    :return: scipy.odr.Output class, with estimated parameters in '.beta'.
    """

    # Create a Model.
    linear_model = odr.Model(linear)
    
    # Create a Data or RealData instance.
    mydata = odr.RealData(x, y, sx=xerrors, sy=yerrors)
    
    # Instantiate ODR with your data, model and initial parameter estimate.
    myodr = odr.ODR(mydata, linear_model, beta0=[0, 0])
    
    # Run the fit.
    myoutput = myodr.run()
    
    return myoutput



def calc_rotation_yoffset(branches, show=False):
    """
    Calculate the rotation of the detector and the offset in y direction from the propagated hits of a given event.
    
    :param prop_x: Propagated hit positions in x direction.
    :param prop_y: Propagated hit positions in y direction.
    :return: The rotation of the detector in radians and the offset in the y direction.
    """
    
    # Data needed for the calculation.
    prop_x, prop_y, rec_x, rec_y, residuals_x, residuals_y, valid_props, valid_recs, chi2_mask = calc_residuals(branches)
    rechit_eta = branches['rechitEta'][chi2_mask][valid_props]#[nohit_mask][notNone_mask]
    botheta_mask = (np.any(rechit_eta == 1, axis=1) & np.any(rechit_eta == 2, axis=1))
    
    # Make bins for two dimensional histogram of propagated x and y positions. 
    # These bins are the ones used for the fit, so limit the ranges.
    minbinx = -25
    maxbinx = 40
    binsizex = 5
    binsx = np.arange(minbinx-binsizex/2, maxbinx+binsizex*3/2, binsizex)

    minbiny = -2
    maxbiny = 1
    binsizey = 0.1
    binsy = np.arange(minbiny-binsizey/2, maxbiny+binsizey*3/2, binsizey)
    
    bincentersx = binsx[:-1] + binsizex/2
    bincentersy = binsy[:-1] + binsizey/2

    # Histogram of the propagated x and y positions.
    h, _, _, _ = plt.hist2d(ak.flatten(prop_x)[botheta_mask], ak.flatten(prop_y)[botheta_mask], bins=[binsx, binsy])

    # Calculate points and errorbars to make the fit.
    xvals = []
    yvals = []
    xerrors = []
    yerrors = []
    for index, i in enumerate(h):
        try:
            # Datapoint.
            mean = np.average(bincentersy, weights=i)
            xvals.append(bincentersx[index])
            yvals.append(mean)

            # For the y axis it's gaussian, so you can use the standard deviation as errors. 
            yerrors.append(np.average((bincentersy - mean)**2, weights=i))
            
            # In the x axis the points are uniformly distributed in each bin, 
            # so you can use either the RMS or the bin size divided by sqrt(12) as errors.
            xerrors.append(binsizex/np.sqrt(12))
            
        except ZeroDivisionError:
            pass

    plt.close()

    # Make bins for two dimensional histogram of propagated x and y positions. 
    # These bins are the ones used for the plot, so large enough to have an overview
    minbinx = -25
    maxbinx = 40
    binsizex = 5
    binsx = np.arange(minbinx-binsizex/2, maxbinx+binsizex*3/2, binsizex)

    minbiny = -10
    maxbiny = 10
    binsizey = 0.25
    binsy = np.arange(minbiny-binsizey/2, maxbiny+binsizey*3/2, binsizey)

    # Plot histogram and mean values and fit a line through the points.
    fig, ax = plt.subplots()
    h = ax.hist2d(ak.flatten(prop_x)[botheta_mask], ak.flatten(prop_y)[botheta_mask], bins=[binsx, binsy])
    ax.errorbar(xvals, yvals, yerrors, xerrors, 'r.', elinewidth=0.6)

    myoutput = fit_line(xvals, yvals, xerrors, yerrors)
    ax.plot(np.linspace(-40, 40, 1000), linear(myoutput.beta, np.linspace(-40, 40, 1000)), color='red')

    if show:
        print(f"The tilt of the line is {np.arctan(myoutput.beta[0]):.5f} radians")
        print(f"The offset of the line is {myoutput.beta[1]:.5f}mm")
        ax.set(xlabel="Prop x", ylabel="Prop y", title="Rotation of the detector")
        fig.colorbar(h[3], label="Number of hits")
        plt.show()

    plt.close()
   
    return np.arctan(myoutput.beta[0]), myoutput.beta[1]