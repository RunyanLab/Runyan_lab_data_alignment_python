import numpy as np
import os 
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import warnings
from scipy import stats
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ranksums
from scipy.signal import find_peaks
from numba import jit, prange
import matplotlib as mpl
import random
import pickle
import gzip
import pyabf
from pathlib import Path
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
from oasis.plotting import simpleaxis
from oasis.oasis_methods import oasisAR1, oasisAR2
import hdf5storage
from neo.io import AxonIO
import platform
from concurrent.futures import ProcessPoolExecutor
from matplotlib import cm

class deconvolution:
    """
    A class for performing deconvolution operations on neural imaging data.
    
    This class provides methods for computing DFF (Delta F/F), deconvolving signals,
    and handling large MAT files for neural data processing.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_percentiles(size, fReflect, fReflect2, numFrames, cellCount):
        """
        Just-in-time compiled function to compute percentiles in parallel.
        
        Args:
            size (int): Window size for percentile computation
            fReflect (ndarray): Reflected array of neural activity data
            fReflect2 (ndarray): Second reflected array of neural activity data
            numFrames (int): Number of frames in the data
            cellCount (int): Number of cells/neurons
            
        Returns:
            tuple: (resultArray, resultArray2)
                - resultArray (ndarray): 8th percentile values for first array
                - resultArray2 (ndarray): 8th percentile values for second array
        """
        resultArray = np.zeros((cellCount, numFrames))
        resultArray2 = np.zeros((cellCount, numFrames))
        for c in prange(cellCount):
            for frame in range(size, numFrames + size):
                tempArray = fReflect[c, (frame-size):frame+size]
                tempArray2 = fReflect2[c, (frame-size):frame+size]

                resultArray[c, frame-size] = np.percentile(tempArray, 8)
                resultArray2[c, frame-size] = np.percentile(tempArray2, 8)
        return resultArray, resultArray2

    @staticmethod
    def load_large_mat_file(filepath, variable_names=None):
        """
        Loads specific variables from a .mat file efficiently.
        
        Args:
            filepath (str): Path to the .mat file
            variable_names (list, optional): List of variable names to load
            
        Returns:
            dict: Dictionary with keys as variable names and values as numpy arrays
            
        Raises:
            MemoryError: If insufficient memory to load the file
        """

        try:
            if variable_names:
                mat_data = scipy.io.loadmat(filepath, variable_names=variable_names)
            else:
                mat_data = scipy.io.loadmat(filepath)
            return mat_data
        except MemoryError:
            print("Failed to load due to memory error. Consider loading fewer variables or using a machine with more RAM.")
            return None

    def pydff(self, mouseID, date, server, size, F_file, variables_to_load, AB_or_JM,info):
        """
        Calculate DFF (Delta F/F) and Z-scored DFF from neural imaging data.
        
        Args:
            mouseID (str): Identifier for the mouse
            date (str): Date of the experiment
            server (str): Server name where data is stored
            size (int): Window size for computation
            F_file (str): Fluorescence data file
            variables_to_load (list): Variables to load from the data
            AB_or_JM (str): Identifier for data source ('AB' or 'JM')
            
        Returns:
            tuple: (dff, z_dff)
                - dff (ndarray): Delta F/F values
                - z_dff (ndarray): Z-scored Delta F/F values
        """

        
        directory = Path(f"{info['server']}/{info['experimenter_name']}/ProcessedData/{mouseID}/{date}/suite2p/plane0")
        F = np.load(os.path.join(directory,"F.npy"))
        Fneu =  np.load(os.path.join(directory,"Fneu.npy"))
        iscell = np.load(os.path.join(directory,"iscell.npy"))
    
        print(directory)
        os.chdir(directory)

        # being reset for data incase there is more than one type of cell selection
        # e.g. if red cell selection was done last, iscell.npy will only have the red cells marked as cells 
        # Fall.mat should have all of the cells correctly marked 
        Fall = scipy.io.loadmat(os.path.join(directory,"Fall.mat"))
        iscell = Fall['iscell']

        print("loaded Fall")    
        numROI = F.shape[0]
        icel = np.where(iscell[:, 0] == 1)[0]
        count = len(icel)
        fBodyRawPre = F[icel, :].astype(float)
        fPre = (F[icel, :] - 0.7 * Fneu[icel, :]).astype(float)

        numFrames = fBodyRawPre.shape[1]
        fSom = fPre
        fBodyRawSom = fBodyRawPre

        fReflect = np.concatenate((np.fliplr(fSom[:, :size]), fSom, np.fliplr(fSom[:, -size:])), axis=1)
        fReflect2 = np.concatenate((np.fliplr(fBodyRawSom[:, :size]), fBodyRawSom, np.fliplr(fBodyRawSom[:, -size:])), axis=1)

        resultArray, resultArray2 = self.compute_percentiles(size, fReflect, fReflect2, numFrames, count)

        dff = (fSom - resultArray) / resultArray2
        z_dff = scipy.stats.zscore(dff, axis=1)

        if AB_or_JM == 'AB':
            directory = Path(f"/Volumes/{server}/Akhil/ProcessedData/{mouseID}/{date}")
        else:
            directory = Path(f"/Volumes/{server}/Jordyn/Processed Data/{mouseID}/{date}")
        directory.mkdir(parents=True, exist_ok=True)

        return dff, z_dff, icel

    @staticmethod
    def decovolve(dff):
        """
        Perform deconvolution on DFF signals using OASIS method.
        
        Args:
            dff (ndarray): Delta F/F signal array of shape (neurons x timepoints)
            
        Returns:
            tuple: (denoisesig, deconvsig, dff_interp, options)
                - denoisesig (ndarray): Denoised signal
                - deconvsig (ndarray): Deconvolved signal
                - dff_interp (ndarray): Interpolated DFF
                - options (ndarray): Lambda values used in deconvolution
        """
        dff_interp = [] 
        denoisesig = []
        deconvsig = []
        options = []
        for neuron_number in range(dff.shape[0]):
            c, s, b, g, lam = deconvolve(dff[neuron_number],
                                       g=estimate_parameters(dff[neuron_number], 1)[0],
                                       penalty=1)
            denoisesig.append(c)
            deconvsig.append(s)
            dff_interp.append(g)
            options.append(lam)

        return (np.array(denoisesig), np.array(deconvsig),
                np.array(dff_interp), np.array(options))

    @staticmethod
    def save_as_mat_v73(file_path, data_dict):
        """
        Saves a dictionary of numpy arrays to a .mat file in MATLAB v7.3 format.
        
        Args:
            file_path (str): Path where the .mat file will be saved
            data_dict (dict): Dictionary where keys are variable names and values are numpy arrays
            
        Returns:
            None
            
        Notes:
            Automatically transposes 2D+ arrays to match MATLAB's column-major order
        """
        with h5py.File(file_path, 'w') as file:
            for key, value in data_dict.items():
                if value.ndim > 1:
                    value = value.T
                file.create_dataset(key, data=value)
            file.attrs['MATLAB_class'] = 'double'
            print(f"Data has been saved to {file_path} in MATLAB v7.3 format.")

    def saving_function(self, server, mouseID, date, variable):
        """
        Save variables to a MAT file in a specific directory structure.
        
        Args:
            server (str): Server name where data will be saved
            mouseID (str): Identifier for the mouse
            date (str): Date of the experiment
            variable (ndarray): Data to be saved
            
        Returns:
            None
            
        Notes:
            Saves data in the format: /Volumes/server/Akhil/ProcessedData/mouseID/date/spikes/variable.mat
        """
        directory = Path(f"/Volumes/{server}/Akhil/ProcessedData/{mouseID}/{date}/spikes/{variable}.mat")
        dff_dict = {
            f'{variable}': variable
        }
        self.save_as_mat_v73(directory, dff_dict)

class alignment:
    """
    A class for aligning neural imaging data with behavioral events.
    
    This class provides methods for synchronizing neural activity data with 
    behavioral timestamps and events during experimental trials.
    """
    # Intialization methods 
    def __init__(self):
        pass

    # 'PUBLIC' methods 
    def load_virmen_data(self, info):
        """
        Load and structure VirMEn behavioral data from MAT files.
        
        Args:
            self: Instance of alignment class
            info (dict): Dictionary containing path information with key:
                - virmen_base (str): Base path to VirMEn data files
                
        Returns:
            tuple: (structured_dataCell, data)
                - structured_dataCell (dict): Dictionary containing trial-by-trial data with structured fields
                - data (dict): Raw data from the MAT file
                
        Notes:
            Handles different file naming conventions (_Cell_2.mat, _Cell_1.mat, or _Cell.mat)
            and extracts detailed trial information including:
            - Mouse and experiment metadata
            - Trial results and timing
            - Maze conditions and behavioral outcomes
        """
        # Load data and dataCell based on available files
        data = {}
        dataCell = {}

        if os.path.isfile(f"{info['virmen_base']}_Cell_2.mat"):
            data = loadmat(f"{info['virmen_base']}_2.mat")
            dataCell = loadmat(f"{info['virmen_base']}_Cell_2.mat")
        elif os.path.isfile(f"{info['virmen_base']}_Cell_1.mat"):
            data = loadmat(f"{info['virmen_base']}_1.mat")
            dataCell = loadmat(f"{info['virmen_base']}_Cell_1.mat")
        else:
            data = loadmat(f"{info['virmen_base']}.mat")
            dataCell = loadmat(f"{info['virmen_base']}_Cell.mat")

        # Convert dataCell to a structured dictionary
        structured_dataCell = {}
        raw_dataCell = dataCell["dataCell"][0]

        for trial_idx, trial in enumerate(raw_dataCell):
            trial_info = {}

            # Extract 'info' fields
            info_data = trial["info"].item()
            trial_info["mouse"] = info_data["mouse"].item()
            trial_info["date"] = info_data["date"].item()
            trial_info["experimenter"] = info_data["experimenter"].item()
            trial_info["conditions"] = info_data["conditions"].item()
            trial_info["itiCorrect"] = int(info_data["itiCorrect"].item())
            trial_info["itiMiss"] = int(info_data["itiMiss"].item())
            trial_info["name"] = info_data["name"].item()
            trial_info["whiteMazes"] = info_data["whiteMazes"].item()
            trial_info["leftMazes"] = info_data["leftMazes"].item()

            # Extract 'result' fields
            result_data = trial["result"].item()
            trial_info["correct"] = int(result_data["correct"].item())
            trial_info["rewardSize"] = int(result_data["rewardSize"].item())
            trial_info["leftTurn"] = int(result_data["leftTurn"].item())
            trial_info["whiteTurn"] = int(result_data["whiteTurn"].item())
            trial_info["streak"] = int(result_data["streak"].item())

            # Extract 'time' fields
            time_data = trial["time"].item()
            trial_info["start"] = float(time_data["start"].item())
            trial_info["stop"] = float(time_data["stop"].item())
            trial_info["durationMatlabUnits"] = float(time_data["durationMatlabUnits"].item())
            trial_info["duration"] = float(time_data["duration"].item())
            trial_info["sessionStart"] = float(time_data["sessionStart"].item())

            # Extract 'maze' fields
            maze_data = trial["maze"].item()
            trial_info["condition"] = int(maze_data["condition"].item())
            trial_info["leftTrial"] = int(maze_data["leftTrial"].item())
            trial_info["whiteTrial"] = int(maze_data["whiteTrial"].item())
            trial_info["is_stim_trial"] = int(maze_data["is_stim_trial"].item()) if "is_stim_trial" in maze_data.dtype.names else np.NaN

            # Add to structured_dataCell dictionary
            structured_dataCell[f"trial_{trial_idx + 1}"] = trial_info

        return structured_dataCell, data  # Return as structured dictionaries

    def align_virmen_iterations_to_digidata(self, base, virmen_channel):
        """
        Extract virmen iterations and timing information from digidata files using positive peaks.
        Return the location of each VIRMEN iteration in pClamp time broken into the corresponding pClamp file
        
        Args:
            self: Instance of alignment class
            base (str): Base directory path containing sync files
            virmen_channel_number (int): Channel number for virmen data
            
        Returns:
            tuple: (acquisitions, updated_trial_its, sound_condition_array)
                - acquisitions (list): List of dictionaries containing processed data for each file
                - updated_trial_its (dict): Updated trial iteration information
                - sound_condition_array (list): Updated sound condition information
        """

        # clear the plots
        plt.figure(21,figsize=[15,6])
        plt.clf()
        
        plt.figure(22,figsize=[15,6])
        plt.clf()

        # Get list of sync files
        sync_files = [f for f in os.listdir(base) if f.endswith('.abf')]

        # Initalize the acquisition structure 
        acquisitions = []

        for file_ind, file in enumerate(sync_files):
            sync_file_path = os.path.join(base, file)

            # Load sync data
            sync_data, sampling_rate = self.load_sync_data(sync_file_path)

            # Process temp signal
            iteration_signal = sync_data[:,virmen_channel]
            iteration_signal = iteration_signal - np.median(iteration_signal)

            # Find peaks with adjusted parameters
            peaks = find_peaks(np.abs(iteration_signal), 
                                            height=0.09,
                                            distance=15)[0]

            if len(peaks) == 0:
                print(f"No peaks found in file {file}")
                # continue
                
            # Normalize peaks
            iteration_signal = iteration_signal / np.abs(np.mean(iteration_signal[peaks]))

            # Adjust peak locations by adding 1 
            # Why is this being done? 
            # peaks = peaks + 1

            # Iteration values are the peaks, the times are the indexes
            # it_times is indexed relative to pclamp timing
            it_values = iteration_signal[peaks]
            it_times = peaks

            # Find positive peaks
            # positive peaks are used to align pclamp time with time in virmen
            # positive peaks occur every 10,000 iterations and increase in height by .1 each positive peak
            marked_its = np.where(it_values > 0)[0]
            if len(marked_its) == 0:
                print(f"No positive peaks found in file {file}")
                # continue

            # actual_it_values will be relative to the total number of iterations 
            # in the matlab file virmen saves as 'data'
            actual_it_values = np.zeros(len(it_values))

            # Process first iteration with adjusted rounding
            # if first iteration is an abnormal size like 10 
            if round(it_values[marked_its[0]], 0) == 10:
                actual_it_values[marked_its[0]] = 0
            else:
                # find first positive peak, use that value in terms of a multiple of 10k to set the absolute iteration
                # indexes for the previous frames
                actual_it_values[marked_its[0]] = round((it_values[marked_its[0]] * 1e5) / 1e4) * 1e4
                # Process previous iterations
                next_value = actual_it_values[marked_its[0]]
                for it in range(marked_its[0] - 1, -1, -1):
                    actual_it_values[it] = next_value - 1
                    next_value = actual_it_values[it]

            # use the positive peak to asign absolute iteration indexes for the rest of the peaks in the pclamp data   
            previous_value = actual_it_values[marked_its[0]]
            for it in range(marked_its[0] + 1, len(it_values)):
                actual_it_values[it] = previous_value + 1
                previous_value = actual_it_values[it]


            # Store results
            acquisition = {
                'actual_it_values': actual_it_values,
                'it_times':  it_times,
                'positive_peaks' : marked_its,
                'positive_peaks_values' : actual_it_values[marked_its],
                'directory': sync_files[file_ind]
            }
                
            acquisitions.append(acquisition)

            # Optional plotting
            plt.figure(21,figsize=[15,6])
            plt.subplot(4,4,file_ind+1)

            # make subplots, organize by the number of acquisitions 
            plt.xlabel('Time (kHz)')
            plt.ylabel('Iterations')
            plt.title('Aquisition - ' + str(file_ind))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.plot(iteration_signal,color='gray')
            # plt.plot(it_times[marked_its], np.zeros_like(marked_its), color='red')
            y_vals = sync_data[it_times[marked_its],virmen_channel]
            plt.scatter(it_times[marked_its], y_vals, color='red')

            plt.tight_layout()
        
        plt.figure(21,figsize=[15,6])
        plt.suptitle('Virmen Iteration Positive Peaks',fontweight='bold')
        plt.tight_layout()
        plt.show()
        # Final plot
        plt.figure(22)
        plt.title('Virmen Iteration Values Across Files',fontweight='bold')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Iteration times (pclamp time)')
        plt.ylabel('Iteration values (virmen data values)')
        plt.tight_layout()

        jet_cmap = cm.get_cmap('jet',len(sync_files)*3)
        jet_colors = jet_cmap(np.linspace(0,1,len(sync_files)*3))
        for ind, acq in enumerate(acquisitions):
            plt.plot(acq['it_times'], acq['actual_it_values'],color=jet_colors[ind,:])
        plt.show()

        marked_list = []
        for acq in acquisitions:
            marked_list.append(acq['positive_peaks_values'])
        flattened_array = np.concatenate(marked_list)

        if not any(np.diff(flattened_array) == 10000):
            print('ERROR : Missing or additional virmen iterations. Cannot align VIRMEN data with imaging data')
            raise ValueError("VIRMEN Iterations NOT aligned")
        else:
            print('Positive peaks in VIRMEN are separated by 10k iterations. VIRMEN data successfully aligned')

        plt.figure()
        plt.title('Positive Peak Values',fontweight='bold')
        x = np.arange(1,len(flattened_array)+1)
        plt.bar(x,flattened_array,color='black')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Positive Peak Number')
        plt.ylabel('Iteration Value')

        for ii, value in enumerate(flattened_array):
            plt.text(ii,value + .10, f'{value//1000}k',ha='center',va='bottom')

        plt.tight_layout()
        plt.show()



        return acquisitions   # Replace None with updated_trial_its if needed

    def get_frame_times(self, imaging_base_path, sync_base_path, channel_number, plot_on):
        """
        Extract frame timing information from synchronization files for imaging data.
        
        Args:
            self: Instance of alignment class
            imaging_base_path (str): Path to the imaging data directory
            sync_base_path (str): Path to the synchronization files directory
            channel_number (int): Channel number for sync signal
            plot_on (bool): Whether to display diagnostic plots
            
        Returns:
            list: List of dictionaries containing alignment information for each acquisition:
                - imaging_id (str): TSeries folder name
                - sync_id (str): Sync file name
                - sync_sampling_rate (float): Sampling rate of sync signal
                - frame_times (ndarray): Array of frame timestamps
                
        Raises:
            ValueError: If number of TSeries folders doesn't match number of sync files
            
        Notes:
            Supports both .abf and .h5 sync file formats
            Detects frames using peak detection on galvo signals
            Optionally displays diagnostic plots of signal and frame detection
        """
        alignment_info = []

        # Determine if the sync files are in .abf or .h5 format
        sync_files = [f for f in os.listdir(sync_base_path) if f.endswith('.abf') or f.endswith('.h5')]
        num_syncs = len(sync_files)
        is_pclamp = any(f.endswith('.abf') for f in sync_files)

        # Ensure number of TSeries matches the number of syncs
        tseries_folders = [f for f in os.listdir(imaging_base_path) if 'TSeries' in f]
        num_tseries = len(tseries_folders)
        if num_syncs != num_tseries:
            raise ValueError("Number of TSeries does not match Number of Syncs")

        # Get the number of TIF files per TSeries folder
        num_tifs = [len([f for f in os.listdir(os.path.join(imaging_base_path, folder)) if 'Ch2' in f]) for folder in tseries_folders]

        for acq_number in range(num_syncs):
            print(f"Imaging Dir: {tseries_folders[acq_number]}, Sync File: {sync_files[acq_number]}")
            sync_file_path = os.path.join(sync_base_path, sync_files[acq_number])
            imaging_folder = tseries_folders[acq_number]

            # Load sync data based on file type
            if is_pclamp:
                # reader = AxonIO(sync_file_path)
                # segments = reader.read_block().segments[0]
                #if channel_number >= segments.analogsignals[0].shape[1]:
                #    raise IndexError(f"Channel number {channel_number} is out of bounds for available channels.")
                sync_data, sync_sampling_rate = self.load_sync_data(sync_file_path)
                sync_data = sync_data[:,channel_number]
                # sync_data = segments.analogsignals[2][:, channel_number].magnitude.flatten()
                # sync_sampling_rate = float(segments.analogsignals[0].sampling_rate)

            galvo_signal_norm = sync_data / np.max(sync_data)
            peaks, _ = find_peaks(galvo_signal_norm, height=0.3, prominence=0.1)
            scanning_amplitude = np.mean(sync_data[peaks])
            good_peaks = [p for p in peaks if 0.95 * scanning_amplitude < sync_data[p] < 1.1 * scanning_amplitude]
            frame_times = np.array(good_peaks)

            # Calculate frame rate and intervals
            frame_intervals_sec = np.diff(frame_times) / sync_sampling_rate
            imaging_frame_rate = 1 / (np.mean(frame_intervals_sec))
            alignment_info.append({
                "imaging_id": imaging_folder,
                "sync_id": sync_files[acq_number],
                "sync_sampling_rate": sync_sampling_rate,
                "frame_times": frame_times
            })

            # Optionally plot the galvo signal and frame detection
            if plot_on:
                plt.figure(figsize=(10, 5))
                plt.plot(sync_data, label="Galvo Signal")
                plt.plot(frame_times, sync_data[frame_times], 'r*', label="Frame Times")
                plt.title("Galvo Signal and Defined Frame Times")
                plt.legend()
                plt.show()

                plt.figure()
                plt.hist(np.diff(frame_times))
                plt.title("Unique Frame Intervals")
                plt.show()

            # Adjust frame times if necessary based on TIF file count
            if len(frame_times) > num_tifs[acq_number]:
                frame_times = frame_times[:num_tifs[acq_number]]

            alignment_info[acq_number]["frame_times"] = frame_times

        return alignment_info

    def get_frame_times_parallel(self, imaging_base_path, sync_base_path, channel_number, plot_on):
        """
        Extract frame timing information from synchronization files for imaging data.
        
        Args:
            self: Instance of alignment class
            imaging_base_path (str): Path to the imaging data directory
            sync_base_path (str): Path to the synchronization files directory
            channel_number (int): Channel number for sync signal
            plot_on (bool): Whether to display diagnostic plots
            
        Returns:
            list: List of dictionaries containing alignment information for each acquisition:
                - imaging_id (str): TSeries folder name
                - sync_id (str): Sync file name
                - sync_sampling_rate (float): Sampling rate of sync signal
                - frame_times (ndarray): Array of frame timestamps
                
        Raises:
            ValueError: If number of TSeries folders doesn't match number of sync files
            
        Notes:
            Supports both .abf and .h5 sync file formats
            Detects frames using peak detection on galvo signals
            Optionally displays diagnostic plots of signal and frame detection
        """
        alignment_info = []

        # Determine if the sync files are in .abf or .h5 format
        sync_files = [f for f in os.listdir(sync_base_path) if f.endswith('.abf') or f.endswith('.h5')]
        num_syncs = len(sync_files)
        is_pclamp = any(f.endswith('.abf') for f in sync_files)

        # Ensure number of TSeries matches the number of syncs
        tseries_folders = [f for f in os.listdir(imaging_base_path) if 'TSeries' in f]
        num_tseries = len(tseries_folders)
        if num_syncs != num_tseries:
            raise ValueError("Number of TSeries does not match Number of Syncs")

        # Get the number of TIF files per TSeries folder
        num_tifs = [len([f for f in os.listdir(os.path.join(imaging_base_path, folder)) if 'Ch2' in f]) for folder in tseries_folders]

        with ProcessPoolExecutor() as executor:
            results = executor.map(self.process_aquisition,range(num_syncs),
                                   [tseries_folders]*num_syncs,
                                   [sync_files] * num_syncs,
                                   [sync_base_path] * num_syncs,
                                   [is_pclamp]*num_syncs,
                                   [channel_number] * num_syncs,
                                   [num_tifs] * num_syncs,
                                   [plot_on] * num_syncs)
        
        alignment_info = list(results)
        if plot_on:
            plt.figure(figsize=(15, 6))
            num_plots = int(np.round(np.sqrt(len(alignment_info))))

            for ind , acq in enumerate(alignment_info):
                plt.subplot(num_plots,num_plots,ind+1)
                plt.plot(acq['sync_data'], label="Galvo Signal", color='mediumseagreen')
                plt.scatter(acq['frame_times'], acq['sync_data'][acq['frame_times']], color='red', label="Frame Times")
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.title('Acquisition - ' + str(ind))
                plt.ylabel('Voltage')
                plt.xlabel('Time 10kHz')
                plt.tight_layout()
            plt.suptitle("Galvo Signal and Defined Frame Times",fontweight='bold')


            plt.tight_layout()
            plt.show()


            plt.figure(figsize=(15, 6))
            for ind, acq in enumerate(alignment_info):
                plt.subplot(num_plots,num_plots,ind+1)
                plt.title('Acquisition - ' + str(ind))
                plt.hist(np.diff(acq['frame_times']),color='mediumseagreen')
                plt.xlabel('Time between frames')
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            plt.suptitle("Unique Frame Intervals",fontweight='bold')
            plt.tight_layout()
            plt.show()


        return alignment_info
        
    def align_virmen_data(self, dff, deconv, virmen_aq, alignment_info, data, dataCell):
        """
        Align VirMEn behavioral data with imaging data and organize into trial-by-trial format.
        
        Args:
            self: Instance of alignment class
            dff (ndarray): Delta F/F calcium imaging data (neurons x time)
            deconv (ndarray): Deconvolved calcium imaging data (neurons x time)
            virmen_aq (list): List of dictionaries containing VirMEn acquisition data
            alignment_info (list): List of dictionaries with frame timing information
            data (dict): Raw VirMEn data dictionary
            dataCell (dict): Processed VirMEn data organized by trials
        
        Returns:
            dict: Dictionary where each key is 'trial_X' containing:
                - start_it: Trial start iteration
                - end_it: Trial end iteration
                - iti_start_it: ITI start iteration
                - iti_end_it: ITI end iteration
                - virmen_trial_info: Dictionary of trial behavioral data
                - frame_id: Imaging frame IDs for this trial
                - dff: Delta F/F data for this trial
                - z_dff: Z-scored Delta F/F data for this trial
                - deconv: Deconvolved data for this trial
                - relative_frames: Frame indices relative to session start
                - file_num: Recording file number
                - movement_in_virmen_time: Movement data in VirMEn time
                - movement_in_imaging_time: Movement data aligned to imaging frames
                - good_trial: Flag indicating valid trial (1) or not (0)
        """
        # Z-score dff (assumes neurons x time)
        z_dff = scipy.stats.zscore(dff, axis=1)
        
        # Extract the data array from the MATLAB structure
        virmen_data = data['data']
        
        # Initialize the imaging dictionary
        imaging = {}
        
        # Previous frames are used to index properly into dff and deconv traces
        # which have a length corresponding to the amount of imaged frames and are 
        # not broken into individual acquisitions
        file_ind = 0
        previous_frames_sum = 0
        previous_frames_temp = 0
        previous_frames = 0

        
        # Get number of trials
        num_trials = len(dataCell)
        
        # Iterate through trials
        for vr_trial in range(1, num_trials):
            current_trial = f'trial_{vr_trial}'
            next_trial = f'trial_{vr_trial + 1}'
            
            # Get trial start and end times
            start_time = dataCell[current_trial]['start']
            stop_time = dataCell[current_trial]['stop']
            
            # Get next trial start time if it exists
            if next_trial in dataCell:
                next_start_time = dataCell[next_trial]['start']
            else:
                continue
            
            # Find corresponding iterations for trial start and end points in VIRMEN data
            start_it = np.argmin(np.abs(virmen_data[0, :] - start_time)) + 1
            end_it = np.argmin(np.abs(virmen_data[0, :] - stop_time)) + 1
            iti_end_it = np.argmin(np.abs(virmen_data[0, :] - next_start_time))
            iti_start_it = end_it + 1
        
            
            # Initialize the trial dictionary
            imaging[current_trial] = {
                'start_it': start_it,
                'end_it': end_it,
                'iti_start_it': iti_start_it,
                'iti_end_it': iti_end_it,
                'virmen_trial_info': {
                    'correct': dataCell[current_trial]['correct'],
                    'left_turn': dataCell[current_trial]['leftTurn'],
                    'condition': dataCell[current_trial]['condition']
                },
                'dff': np.NaN,
                'z_dff': np.NaN,
                'deconv': np.NaN,
                'relative_frames': np.NaN,
                'file_num': np.NaN,
                'movement_in_virmen_time': np.NaN,
                'frame_id': np.NaN,
                'movement_in_imaging_time': np.NaN,
                'good_trial': np.NaN
            }
            
            # Check if trial is within bounds
            # Look in current sync file bounds for the absolute iterations which correspond to the current trial
            # If they exist within this file, fill in the imaging structure
            if (start_it >= virmen_aq[file_ind]['actual_it_values'][0] and 
                end_it < virmen_aq[file_ind]['actual_it_values'][-1]):
                
                # Create output data dictionary
                output_data = {
                    'frame_times': alignment_info[file_ind]['frame_times'], # frame times in pclamp time 
                    'iteration_times': virmen_aq[file_ind]['it_times'], # iteration times in pclamp time 
                    'iteration_ids': virmen_aq[file_ind]['actual_it_values'] # iteration values relative to virmen 'data' matrix 
                }
                
                # Check if trial full length of the trial is within frame limits
                # sync files should contain virmen iterations for the full length of the file, but not neccessarily frames
                if (iti_end_it <= output_data['iteration_ids'][-1] and
                    any(output_data['iteration_times'][output_data['iteration_ids'] == start_it] >= output_data['frame_times'][0]) and
                    any(output_data['iteration_times'][output_data['iteration_ids'] == end_it] <= alignment_info[file_ind]['frame_times'][-1])):
                    
                    # Get movement data from the virmen data matrix 
                    # movement_data = self.get_movement_data(virmen_data, start_it, iti_end_it)
                    movement_data = self.get_movement_data(virmen_data, start_it, iti_end_it)

                    imaging[current_trial]['movement_in_virmen_time'] = movement_data
                    
                    # Get frame IDs corresponding to virmen iterations
                    # frame_ids = self.get_frame_ids(output_data, start_it, iti_end_it)
                    frame_ids = self.get_frame_ids(output_data, start_it, iti_end_it)

                    if len(frame_ids) == 0:
                        print(f"Skipping trial {vr_trial} due to missing frame IDs")
                        continue
                        
                    imaging[current_trial]['frame_id'] = frame_ids
                    
                    # Get neural activity
                    maze_start_frame = frame_ids[0]
                    iti_end_frame = frame_ids[-1]
                    
                    # Update trial dictionary with neural data
                    imaging[current_trial].update({
                        'dff': dff[:, maze_start_frame + previous_frames_sum:iti_end_frame + previous_frames_sum],
                        'z_dff': z_dff[:, maze_start_frame + previous_frames_sum:iti_end_frame + previous_frames_sum],
                        'deconv': deconv[:, maze_start_frame + previous_frames_sum:iti_end_frame + previous_frames_sum],
                        'relative_frames': np.arange(maze_start_frame + previous_frames_sum, iti_end_frame + previous_frames_sum + 1), ## ISSUE HERE?? WHY PLUS 1??
                        'file_num': file_ind
                    })
                    
                    # Add movement in imaging time
                    # movement_imaging = self.align_movement_to_imaging(movement_data, frame_ids, imaging[current_trial])
                    movement_imaging = self.align_movement_to_imaging(movement_data, frame_ids, imaging[current_trial])

                    imaging[current_trial]['movement_in_imaging_time'] = movement_imaging
                    
                    imaging[current_trial]['good_trial'] = 1

            # continue if we have not yet reached trials within the current sync file        
            elif start_it < virmen_aq[file_ind]['actual_it_values'][0]:
                continue
                
            elif file_ind < len(virmen_aq) - 1:
                file_ind += 1
                if file_ind == 0:
                    previous_frames_temp = 0
                else:
                    previous_frames_temp = len(alignment_info[file_ind - 1]['frame_times'])
                previous_frames = np.append(previous_frames, previous_frames_temp)
                previous_frames_sum = np.sum(previous_frames)
        
        return imaging

    def sound_alignment_imaging(self, imaging, alignment_info, sync_base_path,speaker_1,speaker_2):
        # NEED TO ADD SECOND SPEAKER ALIGNMENT IF SPEAKER OUTPUT IS RECORDED
        files = [f for f in os.listdir(sync_base_path) if f.endswith('.abf')]
        file_num = 0
        sync_data,sampling_rate = self.load_sync_data(os.path.join(sync_base_path,files[file_num]))
        frame_times = np.asarray(alignment_info[file_num]['frame_times'])
 

        for trial in imaging:
            current = imaging[trial]
            
            if current['good_trial'] and not np.isnan(current['relative_frames']).any():

                # checking to see if we're still in the same file, update if not
                if not file_num == current['file_num']:
                    file_num = current['file_num']
                    sync_data,sampling_rate = self.load_sync_data(os.path.join(sync_base_path,files[file_num]))
                    frame_times = alignment_info[file_num]['frame_times']

                # if given the channel numbers of the speakers, add sound traces to imaging structure
                if speaker_1 is not None:
                    sound_trace = sync_data[frame_times,speaker_1]
                    imaging[trial]['speaker_1'] = sound_trace[current['relative_frames']%10000]
                if speaker_2 is not None:
                    sound_trace = sync_data[frame_times,speaker_2]
                    imaging[trial]['speaker_2'] = sound_trace[current['relative_frames']%10000]

                # Sound onset should be at 50 virmen units
                temp_onset = (np.floor(current['movement_in_imaging_time']['y_position']) > 48) & (np.floor(current['movement_in_imaging_time']['y_position']) < 51)
                temp_onset = np.where(temp_onset)[0]
                imaging[trial]['sound_onset'] = temp_onset

        return imaging
    
    def align_virmen_data_behavior_only(self, virmen_aq, data, dataCell):
        """
        Align VirMEn behavioral data with imaging data and organize into trial-by-trial format.
        
        Args:
            self: Instance of alignment class
            virmen_aq (list): List of dictionaries containing VirMEn acquisition data
            data (dict): Raw VirMEn data dictionary
            dataCell (dict): Processed VirMEn data organized by trials
        
        Returns:
            dict: Dictionary where each key is 'trial_X' containing:
                - start_it: Trial start iteration
                - end_it: Trial end iteration
                - iti_start_it: ITI start iteration
                - iti_end_it: ITI end iteration
                - virmen_trial_info: Dictionary of trial behavioral data
                - file_num: Recording file number
                - movement_in_virmen_time: Movement data in VirMEn time
                - good_trial: Flag indicating valid trial (1) or not (0)
        """
        
        # Extract the data array from the MATLAB structure
        virmen_data = data['data']
        
        # Initialize the imaging dictionary
        behavior = {}
        
        # Previous frames are used to index properly into dff and deconv traces
        # which have a length corresponding to the amount of imaged frames and are 
        # not broken into individual acquisitions
        file_ind = 0

        # Get number of trials
        num_trials = len(dataCell)
        
        # Iterate through trials
        for vr_trial in range(1, num_trials):
            current_trial = f'trial_{vr_trial}'
            next_trial = f'trial_{vr_trial + 1}'
            
            # Get trial start and end times
            start_time = dataCell[current_trial]['start']
            stop_time = dataCell[current_trial]['stop']
            
            # Get next trial start time if it exists
            if next_trial in dataCell:
                next_start_time = dataCell[next_trial]['start']
            else:
                continue
            
            # Find corresponding iterations for trial start and end points in VIRMEN data
            start_it = np.argmin(np.abs(virmen_data[0, :] - start_time)) + 1
            end_it = np.argmin(np.abs(virmen_data[0, :] - stop_time)) + 1
            iti_end_it = np.argmin(np.abs(virmen_data[0, :] - next_start_time))
            iti_start_it = end_it + 1
        
            
        # Initialize the trial dictionary
            behavior[current_trial] = {
                'start_it': start_it,
                'end_it': end_it,
                'iti_start_it': iti_start_it,
                'iti_end_it': iti_end_it,
                'virmen_trial_info': {
                    'correct': dataCell[current_trial]['correct'],
                    'left_turn': dataCell[current_trial]['leftTurn'],
                    'condition': dataCell[current_trial]['condition']
                },
                'file_num': np.NaN,
                'movement_in_virmen_time': np.NaN,
                'good_trial': np.NaN
            }

            # Check if trial is within bounds
            # Look in current sync file bounds for the absolute iterations which correspond to the current trial
            # If they exist within this file, file in the imaging structure
            if (start_it >= virmen_aq[file_ind]['actual_it_values'][0] and 
                end_it < virmen_aq[file_ind]['actual_it_values'][-1]):          
                
                # Check if trial full length of the trial is within frame limits
                # sync files should contain virmen iterations for the full length of the file, but not neccessarily frames
                
                    
                # Get movement data from the virmen data matrix 
                # movement_data = self.get_movement_data(virmen_data, start_it, iti_end_it)
                movement_data = self.get_movement_data(virmen_data, start_it, iti_end_it)
                behavior[current_trial]['movement_in_virmen_time'] = movement_data


                x1 = np.where(virmen_aq[file_ind]['actual_it_values']==start_it)[0]
                x2 = np.where(virmen_aq[file_ind]['actual_it_values']==iti_end_it)[0]

                if len(x1) >0 and len(x2)>0:
                    pclamp_inds = np.arange(virmen_aq[file_ind]['it_times'][x1[0]],virmen_aq[file_ind]['it_times'][x2[0]])
                    behavior[current_trial]['pclamp_inds'] = pclamp_inds

                    behavior[current_trial]['it_times'] =virmen_aq[file_ind]['it_times'][x1[0]:x2[0]+1]
                    
                    behavior[current_trial]['good_trial'] = 1

                # movement_data_pclamp = align_movement_to_pclamp([], movement_data, pclamp_inds, virmen_aq[file_ind]['it_times'],start_it,iti_end_it)

                # Update trial dictionary
                behavior[current_trial].update({
                    'file_num': file_ind
                })
                    
            # continue if we have not yet reached trials within the current sync file        
            elif start_it < virmen_aq[file_ind]['actual_it_values'][0]:
                continue
                
            elif file_ind < len(virmen_aq) - 1:
                file_ind += 1
                
        return behavior

    # 'PRIVATE' methods (not actually made private, but are only called internally)
    def get_movement_data(self, data, start_it, iti_end_it):
        """
        Extract movement-related data for a specific trial period.
        
        Args:
            self: Instance of alignment class
            data (numpy.ndarray): Raw VirMEn data array containing movement information
            start_it (int): Starting iteration index
            iti_end_it (int): Ending iteration index (including ITI)
            
        Returns:
            dict: Dictionary containing movement data with keys:
                - y_position: Y-coordinate positions
                - x_position: X-coordinate positions
                - view_angle: Viewing angles
                - x_velocity: Velocity along X-axis
                - y_velocity: Velocity along Y-axis
                - cw: Clockwise rotation values
                - is_reward: Reward state indicators
                - in_ITI: Inter-trial interval indicators
        """
        return {
            'y_position': data[2, start_it:iti_end_it + 1],
            'x_position': data[1, start_it:iti_end_it + 1],
            'view_angle': data[3, start_it:iti_end_it + 1],
            'x_velocity': data[4, start_it:iti_end_it + 1],
            'y_velocity': data[5, start_it:iti_end_it + 1],
            'cw': data[6, start_it:iti_end_it + 1],
            'is_reward': data[7, start_it:iti_end_it + 1],
            'in_ITI': data[8, start_it:iti_end_it + 1]
        }

    def get_frame_ids(self, output_data, start_it, end_it):
        """
        Get imaging frame IDs corresponding to VirMEn iterations.
        
        Args:
            self: Instance of alignment class
            output_data (dict): Dictionary containing:
                - frame_times: Array of imaging frame timestamps
                - iteration_times: Array of VirMEn iteration timestamps
                - iteration_ids: Array of VirMEn iteration IDs
            start_it (int): Starting iteration index
            end_it (int): Ending iteration index
            
        Returns:
            numpy.ndarray: Array of frame IDs corresponding to each iteration
            
        Notes:
            - For each iteration, finds the next available imaging frame
            - If no future frames exist, uses the last available frame
            - Returns empty array if no valid frame IDs are found
        """
        frame_ids = []
        for its in range(start_it, end_it + 1):
            # Find matching iteration indices
            matching_iterations = np.where(output_data['iteration_ids'] == its)[0]
            
            if len(matching_iterations) == 0:
                continue
                
            idx = matching_iterations[0]
            
            # Get the current iteration time
            current_time = output_data['iteration_times'][idx]
            
            # Find frames after this iteration
            future_frames = np.where(output_data['frame_times'] > current_time)[0]
            
            if len(future_frames) == 0:
                # If no future frames, use the last available frame
                frame = len(output_data['frame_times']) - 1
            else:
                frame = future_frames[0]
                
            frame_ids.append(frame)
        
        if not frame_ids:
            print(f"Warning: No frame IDs found for iterations {start_it} to {end_it}")
            return np.array([])
            
        return np.array(frame_ids)

    def align_movement_to_imaging(self, movement_data, frame_ids, trial_dict):
        """
        Align movement data to imaging frame timestamps.
        
        Args:
            self: Instance of alignment class
            movement_data (dict): Dictionary containing movement data in VirMEn time
            frame_ids (numpy.ndarray): Array of frame IDs for alignment
            trial_dict (dict): Dictionary containing trial information including DFF data
            
        Returns:
            dict: Dictionary containing movement data aligned to imaging frames with keys:
                - y_position: Y-coordinate positions
                - x_position: X-coordinate positions
                - y_velocity: Velocity along Y-axis
                - x_velocity: Velocity along X-axis
                - view_angle: Viewing angles
                - is_reward: Reward state indicators
                - in_ITI: Inter-trial interval indicators
                - pure_tones: Pure tone indicators
                
        Notes:
            - Creates zero-filled arrays for each movement variable
            - Maps movement data to corresponding imaging frames
            - Uses NaN for frames without corresponding movement data
        """
        num_frames = len(trial_dict['dff'][0])
        movement_imaging = {
            'y_position': np.zeros(num_frames),
            'x_position': np.zeros(num_frames),
            'y_velocity': np.zeros(num_frames),
            'x_velocity': np.zeros(num_frames),
            'view_angle': np.zeros(num_frames),
            'is_reward': np.zeros(num_frames),
            'in_ITI': np.zeros(num_frames),
            'pure_tones': np.zeros(num_frames)
        }
        
        # Fill in movement data
        # transforms movement data into imaging time from virmen 'data' time
        # the behavior values from the first iteration in the bound of a frame 
        # are assigned to that frame. If no iteration corresponds to a frame, 
        # nans are assigned. 
        for frame in range(num_frames):
            idx = np.where(frame_ids - frame_ids[0] + 1 == frame + 1)[0]
            if len(idx) > 0:
                idx = idx[0]
                for key in movement_data.keys():
                    if key in movement_imaging:
                        movement_imaging[key][frame] = movement_data[key][idx]
            else:
                for key in movement_imaging.keys():
                    movement_imaging[key][frame] = np.nan
        
        return movement_imaging

    def load_sync_data(self, handle):

        abf = pyabf.ABF(handle)
        # Get number of channels
        num_channels = abf.channelCount
        sampling_rate = abf.sampleRate

        # Create array to store all channel data
        sync_data = np.zeros((len(abf.sweepY), num_channels))

        # Load data from each channel
        for ch in range(num_channels):
            abf.setSweep(0, channel=ch)
            sync_data[:, ch] = abf.sweepY

        return sync_data,sampling_rate
    
    def process_aquisition(self,acq_number,tseries_folders,sync_files,sync_base_path,is_pclamp,channel_number,num_tifs,plot_on):

        print(f"Imaging Dir: {tseries_folders[acq_number]}, Sync File: {sync_files[acq_number]}")
        sync_file_path = os.path.join(sync_base_path, sync_files[acq_number])
        imaging_folder = tseries_folders[acq_number]

        # Load sync data based on file type
        if is_pclamp:
            sync_data, sync_sampling_rate = self.load_sync_data(sync_file_path)
            sync_data = sync_data[:,channel_number]

        galvo_signal_norm = sync_data / np.max(sync_data)
        peaks, _ = find_peaks(galvo_signal_norm, height=0.3, prominence=0.1)
        scanning_amplitude = np.mean(sync_data[peaks])
        good_peaks = [p for p in peaks if 0.95 * scanning_amplitude < sync_data[p] < 1.1 * scanning_amplitude]
        frame_times = np.array(good_peaks)

        # Calculate frame rate and intervals
        frame_intervals_sec = np.diff(frame_times) / sync_sampling_rate
        imaging_frame_rate = 1 / (np.mean(frame_intervals_sec))
     
        # Adjust frame times if necessary based on TIF file count
        if len(frame_times) > num_tifs[acq_number]:
            frame_times = frame_times[:num_tifs[acq_number]]


        # Optionally plot the galvo signal and frame detection
        
        return {
            "imaging_id": imaging_folder,
            "sync_id": sync_files[acq_number],
            "sync_sampling_rate": sync_sampling_rate,
            "frame_times": frame_times,
            "sync_data" : sync_data
        }


    # UNUSED METHODS, COULD BE HELPFUL FOR DEBUGGING IN THE FUTURE
    # CURRENTLY NOT ACCESSABLE FROM OUTSIDE OF THE CLASS
    def __get_digidata_iterations(self, sync_base_path, string, virmen_channel):
            """
            Extract iteration timing information from DigiData synchronization files.
            
            Args:
                self: Instance of alignment class
                sync_base_path (str): Path to the synchronization files directory
                string (str): String identifier for sync files
                virmen_channel (int): Channel number for VirMEn data
                
            Returns:
                list: List of dictionaries containing iteration information for each file:
                    - locs (ndarray): Locations of absolute peaks
                    - it_gaps (ndarray): Gaps between iterations
                    - sync_sampling_rate (float): Sampling rate of sync signal
                    - directory (str): Full path to sync file
                    - pos_loc (ndarray): Locations of positive peaks
                    - pos_pks (ndarray): Values of positive peaks
                    
            Notes:
                Processes .abf files to extract synchronization signals
                Uses peak detection to identify iteration boundaries
                Rounds peak values for consistent comparison
            """
            digidata_its = []
            # Find all .abf files containing the sync string
            # files = [f for f in os.listdir(sync_base_path) if string in f and f.endswith('.abf')]
            files = [f for f in os.listdir(sync_base_path) if f.endswith('.abf')]

            
            for file_ind, filename in enumerate(files, start=1):
                # Load the synchronization data
                reader = AxonIO(os.path.join(sync_base_path, filename))
                segments = reader.read_block().segments[0]
                # Get the data and transpose it to match MATLAB's orientation
                sync_data = segments.analogsignals[2][:, virmen_channel].magnitude.flatten().T
                sync_sampling_rate = float(segments.analogsignals[0].sampling_rate)
                
                # Convert to double precision (float in Python)
                sync_data = sync_data.astype(float)
                
                # Find peaks in absolute values (equivalent to MATLAB's findpeaks)
                abs_peaks, abs_properties = find_peaks(np.abs(sync_data), 
                                                    height=0.09, 
                                                    distance=5)
                
                # Find positive peaks
                pos_peaks, pos_properties = find_peaks(sync_data, 
                                                    height=0.09, 
                                                    distance=5)
                
                # Calculate gaps between iterations
                it_gaps = np.diff(abs_peaks)
                
                # Store results in dictionary (similar to MATLAB struct)
                digidata_its.append({
                    "locs": abs_peaks,  # Changed from abs_locs to abs_peaks
                    "it_gaps": it_gaps,
                    "sync_sampling_rate": sync_sampling_rate,
                    "directory": os.path.join(sync_base_path, filename),
                    "pos_loc": pos_peaks,  # Changed from pos_locs to pos_peaks
                    "pos_pks": np.round(sync_data[pos_peaks], 1)  # Get actual peak values
                })

            return digidata_its

    def __virmen_it_rough_estimation(self, data):
        """
        Use virmen iterations to get rough estimate of which iterations are specific trial events
        based on stereotyped gaps between them.
        
        Args:
            self: Instance of alignment class
            data (dict): Dictionary containing data fields, specifically data['data'] as numpy array
            
        Returns:
            tuple: (trial_its, trial_its_time) - Dictionaries containing trial iteration info
        """
        # Constants
        threshold = 350  # lowest value for this mouse ~454 in 500 maze
        
        # Convert data to numpy array if it isn't already
        virmen_data = np.array(data['data'])
        
        # Find ITI transitions
        iti_diff = np.diff(virmen_data[8, :])  # Using 8 instead of 9 due to 0-based indexing
        start_iti_its = np.where(iti_diff > 0)[0] + 1
        end_iti_its = np.where(iti_diff < 0)[0]
        start_trial_its = end_iti_its + 1
        
        # Add first trial
        start_trial_its = np.sort(np.append(start_trial_its, [0]))  # Using 0 instead of 1 for Python indexing
        end_trial_its = start_iti_its - 1
        
        # Create trial_its dictionary
        trial_its = {
            'end_trial_its': end_trial_its,
            'start_iti_its': start_iti_its,
            'end_iti_its': end_iti_its
        }
        
        # Get complete trials
        min_trials = min(len(end_trial_its), len(start_trial_its))
        if len(start_trial_its) > min_trials:
            start_trial_its = start_trial_its[:-1]
        trial_its['start_trial_its'] = start_trial_its
        
        # Convert iteration timing info
        it_times = virmen_data[0, :] * 86400  # convert to seconds
        it_time_gaps = np.diff(it_times)
        
        # Find different trial events based on time gaps
        end_trial_idx = np.where(it_time_gaps > 0.8)[0]
        start_iti_idx = end_trial_idx + 1
        incorrect_idx = np.where((it_time_gaps > 0.8) & (it_time_gaps < 0.95))[0]
        correct_idx = np.where(it_time_gaps > 0.95)[0]
        temp_its = np.where((it_time_gaps > 0.25) & (it_time_gaps < 0.51))[0] + 1
        
        # Create trial_its_time dictionary
        trial_its_time = {
            'start_trial_its_time': np.append([it_times[0]], it_times[temp_its]),
            'end_trial_its_time': it_times[end_trial_idx],
            'start_iti_its_time': it_times[start_iti_idx],
            'correct_its_its_time': it_times[correct_idx],
            'incorrect_its_time': it_times[incorrect_idx]
        }
        
        # Process sound triggers
        y_pos = virmen_data[2, :]  # Using 2 instead of 3 due to 0-based indexing
        temp_onset = np.sort(np.concatenate([
            np.where(np.round(np.floor(y_pos)) == 50)[0],
            np.where(np.round(np.floor(y_pos)) == 51)[0],
            np.where(np.round(np.floor(y_pos)) == 52)[0]
        ]))
        
        # Find sound triggers for each trial
        sound_trigger_its = []
        for t in range(min(len(trial_its['start_trial_its']), len(trial_its['end_trial_its']))):
            trial_mask = (temp_onset > trial_its['start_trial_its'][t]) & (temp_onset < trial_its['end_trial_its'][t])
            if np.any(trial_mask):
                sound_trigger_its.append(temp_onset[trial_mask][0])  # Get first occurrence
        
        trial_its['sound_trigger_its'] = np.array(sound_trigger_its)
        
        return trial_its, trial_its_time

    def __virmen_it_rough_estimation_adjusted(self, data, sampling_rate, it_times, it_values):
        """
        Adjust trial iteration estimates using timing information.
        
        Args:
            self: Instance of alignment class
            data (dict): Dictionary containing virmen data
            sampling_rate (float): Sampling rate of the recording
            it_times (numpy.ndarray): Array of iteration times
            it_values (numpy.ndarray): Array of iteration values
        
        Returns:
            tuple: (trial_its, trial_its_time) - Dictionaries containing adjusted trial iteration info
        """
        # Convert inputs to numpy arrays if they aren't already
        iterations_in_time = np.array(it_times)
        it_values = np.array(it_values)
        
        # Find ITI transitions
        iti_diff = np.diff(data['data'][8, :])  # Using 8 instead of 9 for 0-based indexing
        start_iti_its = np.where(iti_diff > 0)[0] + 2  # +2 because it seems one earlier than it should be
        end_iti_its = np.where(iti_diff < 0)[0]
        start_trial_its = end_iti_its + 3
        
        # Add first trial and sort
        start_trial_its = np.sort(np.append(start_trial_its, [0]))  # Using 0 instead of 1 for Python indexing
        end_trial_its = start_iti_its - 1
        
        # Get complete trials
        min_trials = min(len(end_trial_its), len(start_trial_its))
        if len(start_trial_its) > min_trials:
            start_trial_its = start_trial_its[:-1]
        
        # Find gaps in iterations
        time_diffs = np.diff(iterations_in_time)
        large_gaps = np.where(time_diffs > 0.7 * sampling_rate)[0]
        gap_iterations = it_values[large_gaps]
        iteration_threshold = 8  # Used to move ITI iterations to the next gap
        
        # Adjust end trial iterations based on large gaps
        for g in range(len(large_gaps)):
            # Find closest end trial iteration
            current_gap_idx = np.argmin(np.abs(gap_iterations[g] - end_trial_its))
            val = np.min(np.abs(gap_iterations[g] - end_trial_its))
            
            if val < iteration_threshold:
                end_trial_its[current_gap_idx] = gap_iterations[g]
        
        # Find medium gaps (between 70ms and 700ms)
        medium_gaps = np.where((time_diffs > 0.07 * sampling_rate) & 
                            (time_diffs < 0.7 * sampling_rate))[0]
        gap_iterations = it_values[medium_gaps]
        
        # Adjust end ITI iterations based on medium gaps
        for g in range(len(medium_gaps)):
            current_gap_idx = np.argmin(np.abs(gap_iterations[g] - end_iti_its))
            val = np.min(np.abs(gap_iterations[g] - end_iti_its))
            
            if val < iteration_threshold and (gap_iterations[g] - end_iti_its[current_gap_idx]) >= 0:
                end_iti_its[current_gap_idx] = gap_iterations[g]
        
        # Create output dictionaries
        trial_its = {
            'end_trial_its': end_trial_its,
            'start_iti_its': start_iti_its,
            'end_iti_its': end_iti_its,
            'start_trial_its': start_trial_its
        }
        
        # Create trial_its_time (if needed, similar to previous function)
        trial_its_time = {}  # Add time-based calculations if needed
        
        return trial_its, trial_its_time

    def __get_virmen_iterations_and_times_digidata_positive_peaks(self, base, virmen_channel_number, string, 
                                                                    sound_condition_array, data, file_trial_ids, 
                                                                    file_estimated_trial_info):
        """
        Extract virmen iterations and timing information from digidata files using positive peaks.
        
        Args:
            self: Instance of alignment class
            base (str): Base directory path containing sync files
            virmen_channel_number (int): Channel number for virmen data
            string (str): String identifier for sync files
            sound_condition_array (list): Array of sound condition information
            data (dict): Dictionary containing virmen data
            file_trial_ids (list): List of trial IDs for each file
            file_estimated_trial_info (dict): Dictionary of estimated trial information
            
        Returns:
            tuple: (acquisitions, updated_trial_its, sound_condition_array)
                - acquisitions (list): List of dictionaries containing processed data for each file
                - updated_trial_its (dict): Updated trial iteration information
                - sound_condition_array (list): Updated sound condition information
        """
        acquisitions = []
        
        # Get list of sync files
        # sync_files = [f for f in os.listdir(base) if string in f and f.endswith('.abf')] # akhil method
        sync_files = [f for f in os.listdir(base) if f.endswith('.abf')]

        num_files = len(sync_files)

        for n in range(num_files):
            file_ind = n
            sync_file_path = os.path.join(base, sync_files[n])
            
            # Load sync data
            reader = AxonIO(sync_file_path)
            segments = reader.read_block().segments[0]
            #sync_data = segments.analogsignals[2][:, 1].magnitude.flatten()
            sync_data = segments.analogsignals[2][:, 3].magnitude.flatten()
            sync_sampling_rate = float(segments.analogsignals[0].sampling_rate)
            
            # Process temp signal
            temp = sync_data
            temp = temp - np.median(temp)
            
            # Find peaks with adjusted parameters
            peaks, properties = find_peaks(np.abs(temp), 
                                         height=0.09,
                                         distance=5)
            
            if len(peaks) == 0:
                print(f"No peaks found in file {sync_files[n]}")
                continue
                
            # Normalize peaks
            temp = temp / np.abs(np.mean(temp[peaks]))
            
            # Adjust peak locations by adding 1
            #peaks = peaks + 1
            
            it_values = temp[peaks]
            it_times = peaks

            # Find positive peaks
            marked_its = np.where(it_values > 0)[0]
            if len(marked_its) == 0:
                print(f"No positive peaks found in file {sync_files[n]}")
                continue

            # Initialize actual_it_values
            actual_it_values = np.zeros(len(it_values))
            
            # Process first iteration with adjusted rounding
            if round(it_values[marked_its[0]], 0) == 10:
                actual_it_values[marked_its[0]] = 1 
            else:
                actual_it_values[marked_its[0]] = round((it_values[marked_its[0]] * 1e5) / 1e4) * 1e4
                
                # Process previous iterations
                next_value = actual_it_values[marked_its[0]]
                for it in range(marked_its[0] - 1, -1, -1):
                    actual_it_values[it] = next_value - 1
                    next_value = actual_it_values[it]

            # Process remaining iterations
            previous_value = actual_it_values[marked_its[0]] - 1
            for it in range(marked_its[0] + 1, len(it_values)):
                actual_it_values[it] = previous_value + 1
                previous_value = actual_it_values[it]

            # Initialize possible_it_locs
            if file_trial_ids:
                start_trial_number = file_trial_ids[file_ind][0]
                end_trial_number = file_trial_ids[file_ind][1]
                
                # Get possible iterations within file limits
                #possible_iterations = range(
                #    updated_trial_its['start_trial_its'][start_trial_number],
                #    updated_trial_its['end_iti_its'][end_trial_number] + 1
                #)
                possible_it_locs = np.where(np.isin(actual_it_values))[0]
            else:
                possible_it_locs = None

            # Store results
            acquisition = {
                'actual_it_values': actual_it_values[possible_it_locs] if possible_it_locs is not None else actual_it_values,
                'it_times': it_times[possible_it_locs] if possible_it_locs is not None else it_times,
                'directory': sync_files[file_ind]
            }
            
            if sound_condition_array:
                sound_onsets = sound_condition_array[file_ind]['VR_sounds'][:, 1]
                sound_onsets = sound_onsets[~np.isnan(sound_onsets)]
                acquisition['sound_trigger_time'] = sound_onsets
                
            acquisitions.append(acquisition)

            # Optional plotting
            plt.figure(2)
            plt.clf()
            plt.plot(temp)
            plt.plot(it_times[marked_its], np.zeros_like(marked_its), 'c*')
            plt.pause(0.1)

        # Final plot
        plt.figure(3)
        plt.clf()
        plt.title('virmen iteration values across files')
        for acq in acquisitions:
            plt.plot(acq['it_times'], acq['actual_it_values'])
        plt.show()

        return acquisitions, None, sound_condition_array  # Replace None with updated_trial_its if needed
    

class data_processing:
    """
    A class for processing and analyzing neural imaging data from behavioral experiments.
    
    This class provides methods for converting .mat files to Python dictionaries,
    saving/loading data, and organizing behavioral data into pandas DataFrames.
    """

    # Class-level mapping dictionaries
    visual_stim_maps = {
        'psych': {
            1: 0, 2: 15, 3: 25, 4: 65, 5: 75, 6: 90,
            7: 0, 8: 15, 9: 25, 10: 65, 11: 75, 12: 90,
            13: 0, 14: 15, 15: 25, 16: 65, 17: 75, 18: 90 
        },
        '2loc': {
            1: 0, 2: 90, 3: 0, 4: 90, 5: 90, 6: 0
        },
        'aud': {
            1: np.nan, 2: np.nan
        }
    }
    
    audio_stim_maps = {
        'psych': {
            1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1,
            7: 1, 8: 1, 9: 1, 10: 0, 11: 0, 12: 0,
            13: 1, 14: 1, 15: 1, 16: 0, 17: 0, 18: 0
        },
        '2loc': {
            1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1
        },
        'aud': {
            1: 0, 2: 1
        }
    }
    
    context_maps = {
        'psych': {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
            7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1,
            13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2
        },
        '2loc': {
            1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2
        },
        'aud': {
            1: 2, 2: 2
        }
    }

    def __init__(self):
        pass
        
    def convert_imaging_mat_to_dict(self, path, file_name):
        """
        Convert a MATLAB imaging file to a Python dictionary format.
        
        Args:
            path (str): Path to the directory containing the imaging.mat file
            
        Returns:
            tuple: (data_dict, imaged_trials)
                - data_dict (dict): Dictionary containing processed imaging data
                - imaged_trials (list): List of trial indices that were imaged
        """
        with h5py.File(os.path.join(path, '{}.mat'.format(file_name)), 'r', libver='latest', swmr=True) as f:
            # Get all required data paths upfront to avoid repeated dictionary access
            imaging = f['imaging']
            frame_id = imaging['frame_id']
            start_it = imaging['start_it']
            
            # More efficient trial info collection using numpy operations
            trial_info = np.array([
                np.array(f[frame_id[trial][0]])[0][0] if np.array(f[frame_id[trial][0]])[0] > 0 
                else 0 for trial in range(start_it.shape[0])
            ])
            
            imaged_trials = np.where(trial_info > 1)[0]
            
            # Pre-fetch all data sources
            movement_data = imaging['movement_in_imaging_time']
            behav_data = imaging['virmen_trial_info']
            deconv_data = imaging['deconv']
            dff_data = imaging['dff']
            z_scored_dff_data = imaging['z_dff']
            
            # Define features
            mov_features = ['y_position', 'x_position', 'is_reward', 'in_ITI', 'pure_tones', 
                           'maze_frames', 'iti_frames', 'reward_frames', 'turn_frame']
            behav_features = ['correct', 'left_turn', 'condition']
            
            # Process trials
            data_dict = {}
            for trial in imaged_trials:
                mov_info = f[movement_data[trial][0]]
                behav_info = f[behav_data[trial][0]]
                
                trial_dict = {
                    # Movement features
                    feature: np.array(mov_info[feature], dtype=int).ravel()
                    for feature in mov_features
                }
                
                # Behavior features
                trial_dict.update({
                    feature: int(np.array(behav_info[feature]).ravel()[0])
                    for feature in behav_features
                })
                
                # Neural data
                trial_dict.update({
                    'deconv_data': np.array(f[deconv_data[trial][0]]),
                    'dff_data': np.array(f[dff_data[trial][0]]),
                    'z_dff_data': np.array(f[z_scored_dff_data[trial][0]])
                })
                
                data_dict[trial + 1] = trial_dict
                
            return data_dict, imaged_trials.tolist()
    
    def save_data_dict_as_pickle(self, data_dict, path, mouse_ID, date):
        """
        Save a data dictionary as a compressed pickle file.
        
        Args:
            data_dict (dict): Dictionary containing processed data
            path (str): Directory path where the pickle file will be saved
            mouse_ID (str): Identifier for the mouse
            date (str): Date of the experiment
            
        Returns:
            None
        """
        filepath = os.path.join(path, f"{mouse_ID}_{date}.pickle")
        with gzip.open(filepath, "wb") as file:
            pickle.dump(data_dict, file)
            
    def load_saved_pickle_dict(self, path, mouse_ID, date):
        """
        Load a previously saved compressed pickle file containing processed data.
        
        Args:
            path (str): Directory path where the pickle file is located
            mouse_ID (str): Identifier for the mouse
            date (str): Date of the experiment
            
        Returns:
            dict: Loaded data dictionary
        """
        filepath = os.path.join(path, f"{mouse_ID}_{date}.pickle")
        with gzip.open(filepath, "rb") as file:
            data_dict = pickle.load(file)
        return data_dict

    def convert_array_choice(self, arr):
        """
        Convert choice array to standardized format where 0=Left and 1=Right.
        
        Args:
            arr (numpy.ndarray): Array of choices where 1=Left and 0=Right
            
        Returns:
            list: Converted choices where 0=Left and 1=Right
        """
        return [1 - x for x in arr]
    
    def organize_behavioral_data_into_df(self, data_dict, tasktype='aud'):
        """
        Organize behavioral data into a pandas DataFrame with trial information and event frames.
        
        Args:
            data_dict (dict): Dictionary containing trial data
            tasktype (str): Type of task ('aud', '2loc', or 'psych')
            
        Returns:
            pandas.DataFrame: Organized behavioral data with the following columns:
                - Trial: Trial number
                - trial_type: Type of trial
                - choice: Animal's choice (0=Left, 1=Right)
                - outcome: Trial outcome (0=Incorrect, 1=Correct)
                - Visual_Stim: Visual stimulus value
                - Audio_Stim: Audio stimulus value
                - context: Context value
                - sound onset: Frame when sound starts
                - turn frame: Frame when animal turns
                - iti start: Frame when inter-trial interval starts
                - trial length: Total length of trial in frames
                
        Raises:
            ValueError: If tasktype is not one of 'aud', '2loc', or 'psych'
        """
        if tasktype not in ['aud', '2loc', 'psych']:
            raise ValueError("tasktype must be one of: 'aud', '2loc', 'psych'")

        # Extract trial information
        Conditions = []
        Choices = []
        Outcomes = []
        for trial in data_dict.keys():
            condition = data_dict[trial]['condition']
            choice = data_dict[trial]['left_turn']
            outcome = data_dict[trial]['correct']
            Conditions.append(condition)
            Choices.append(choice)
            Outcomes.append(outcome)

        Conditions = np.array(Conditions)
        Choices = np.array(Choices)
        Outcomes = np.array(Outcomes)

        choice = self.convert_array_choice(Choices)

        # Create initial DataFrame
        task_data = {    
            'Trial': data_dict.keys(),
            'trial_type': Conditions,
            'choice': choice,
            'outcome': Outcomes
        }

        task_df = pd.DataFrame(task_data)

        # Apply mappings based on tasktype
        task_df['Visual_Stim'] = task_df['trial_type'].map(self.visual_stim_maps[tasktype])
        task_df['Audio_Stim'] = task_df['trial_type'].map(self.audio_stim_maps[tasktype])
        task_df['context'] = task_df['trial_type'].map(self.context_maps[tasktype])

        # Add event frames
        task_df = self.task_event_frames(data_dict, task_df)

        # Calculate ITI start frames based on task type
        Iti_start = []
        for trial in task_df.index:
            if tasktype == 'aud':
                if task_df['outcome'][trial] == 0:
                    iti_start = task_df['trial length'][trial] - 75
                else:
                    iti_start = task_df['trial length'][trial] - 45
            else:  # Add specific ITI calculations for other task types if needed
                if task_df['outcome'][trial] == 0:
                    iti_start = task_df['trial length'][trial] - 75
                else:
                    iti_start = task_df['trial length'][trial] - 45
            Iti_start.append(iti_start)
        
        task_df['iti start'] = Iti_start

        # Reorder columns
        task_df = task_df[[
            'Trial',
            'trial_type',
            'choice',
            'outcome',
            'Visual_Stim',
            'Audio_Stim',
            'context',
            'sound onset',
            'turn frame',
            'iti start',
            'trial length'
        ]]

        return task_df

    def task_event_frames(self, data_dict, task_df):
        """
        Extract and add event frame information to the task DataFrame.
        
        Args:
            data_dict (dict): Dictionary containing trial data
            task_df (pandas.DataFrame): DataFrame to add event frames to
            
        Returns:
            pandas.DataFrame: Task DataFrame with added columns:
                - sound onset: Frame when sound starts
                - turn frame: Frame when animal turns
                - trial length: Total length of trial in frames
        """
        sound_onset = []
        turn_frame = []
        iti_start = []
        trial_length = []
        
        for trial in data_dict.keys(): 
            y_pos = data_dict[trial]['y_position'] 
            sound_onset_frame = np.where(y_pos > 50)[0][0]
            turn = data_dict[trial]['turn_frame'][0]
            iti = data_dict[trial]['iti_frames']
            
            sound_onset.append(sound_onset_frame)
            turn_frame.append(turn)
            iti_start.append(iti[0])
            trial_length.append(len(y_pos))
            
        task_df['sound onset'] = sound_onset
        task_df['turn frame'] = turn_frame
        #task_df['iti start'] = iti_start  # Commented out as in original
        task_df['trial length'] = trial_length

        return task_df
    
    def neural_data_extraction(self, data_dict):
        """
        Extract and threshold neural data from trials.
        
        Args:
            data_dict (dict): Dictionary containing trial data
            
        Returns:
            list: List of neural activity matrices for each trial,
                 where each matrix has shape (neurons x frames)
                 Values below 0.05 are set to 0
        """
        trial_data = []  # shape = trials x neurons x frames 
        for trial in data_dict.keys(): 
            neural_data = data_dict[trial]['deconv'].T  # shape = neurons x frames 
            neural_data[neural_data[:,:] < 0.05] = 0 
            trial_data.append(neural_data)
        return trial_data