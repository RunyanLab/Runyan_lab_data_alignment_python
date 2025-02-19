import numpy as np
import os
import glob
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import suite2p as s2p
import h5py
import pyabf
from skimage import measure
import xml.etree.ElementTree as ET
from scipy import signal as sig
from scipy.signal import butter, filtfilt
import ipympl
from neo.io import AxonIO
import hdf5storage
import scipy
import pickle



class GalvoTargetAnalysis:
    def __init__(self):
        # Initialize paths
        self.raw_data_folder = None
        self.suite2p_folder = None
        self.sync_base_path = None
        self.save_folder = None
        self.base = None
        
        # Data containers
        self.stat = None
        self.ops = None
        self.F = None
        self.Fneu = None
        self.spks = None
        self.s2p_badframes = None
        self.nchannels = None  # Added to store number of channels
        self.alignment_info = None  # Store alignment info
        self.dff = None
        self.deconv = None
    
    def get_activity(self, base): 
        # act = os.path.join(base, 'processed_activity_py.pkl')
        # with open (act, "rb") as file: 
        #     activity = pickle.load(file)
        p = os.path.join(base, 'spikes', 'activity_py.mat')
        with h5py.File(p,'r') as f:
            dff = f['dff']
            deconv = f['deconv']
            celi = f['celi']

        self.celi = celi
        self.dff = dff
        self.deconv = deconv

    def get_directories(self, raw_data_path=None, suite2p_path=None, sync_path=None, save_path=None,base=None):
        """Get directories either through parameters or user input"""
        # Use provided paths if available
        self.raw_data_folder = raw_data_path or input("Enter path for 2P Acquisitions directory: ")
        self.suite2p_folder = suite2p_path or input("Enter path for Suite2P directory: ")
        self.sync_base_path = sync_path or input("Enter path for Sync Files directory: ")
        self.save_folder = save_path or input("Enter path for Save directory: ")
        self.base = base 
        
    def load_suite2p_data(self):
        """Load Suite2p data"""
        # Get the ops.npy and stat.npy file paths
        ops_file = os.path.join(self.suite2p_folder, 'ops.npy')
        stat_file = os.path.join(self.suite2p_folder, 'stat.npy')
        
        # Load the files
        self.ops = np.load(ops_file, allow_pickle=True).item()
        self.stat = np.load(stat_file, allow_pickle=True)
        
        # Load F.npy, Fneu.npy, and spks.npy if they exist
        F_file = os.path.join(self.suite2p_folder, 'F.npy')
        Fneu_file = os.path.join(self.suite2p_folder, 'Fneu.npy')
        spks_file = os.path.join(self.suite2p_folder, 'spks.npy')
        iscell_file = os.path.join(self.suite2p_folder, 'iscell.npy')

        
        if os.path.exists(F_file):
            self.F = np.load(F_file)
        if os.path.exists(Fneu_file):
            self.Fneu = np.load(Fneu_file)
        if os.path.exists(spks_file):
            self.spks = np.load(spks_file)
        if os.path.exists(iscell_file):
            self.iscell = np.load(iscell_file)
        
        # Get bad frames
        if 'badframes' in self.ops:
            self.s2p_badframes = np.where(self.ops['badframes'] > 0)[0]
        
    def get_frames_per_folder(self, channel_number=5, plot_on=True):
        """
        Count number of tiff files in each experiment folder to determine frames per folder.
        
        Returns:
            np.ndarray: Array containing the number of frames per folder
        """
        if self.raw_data_folder is None:
            raise ValueError("raw_data_folder not set. Run get_directories() first.")
        if self.nchannels is None:
            raise ValueError("nchannels not set. Please set number of channels first.")
            
        # Use glob pattern matching to find TSeries directories
        tseries_dirs = sorted(glob.glob(os.path.join(self.raw_data_folder, 'TSeries-*')))
        
        # Initialize array to store frame counts
        frame_list = np.zeros(len(tseries_dirs))
        
        # Count tiffs in each directory
        for n, tseries_dir in enumerate(tseries_dirs):
            tif_files = glob.glob(os.path.join(tseries_dir, '*.tif'))
            frame_list[n] = len(tif_files) / float(self.nchannels)
        
        return frame_list
        
    def get_frame_times(self, channel_number=2,frame_times_list=None):
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
        sync_files = [f for f in os.listdir(self.sync_base_path) if f.endswith('.abf') or f.endswith('.h5')]
        num_syncs = len(sync_files)
        is_pclamp = any(f.endswith('.abf') for f in sync_files)

        # Ensure number of TSeries matches the number of syncs
        tseries_folders = [f for f in os.listdir(self.raw_data_folder) if 'TSeries' in f]
        num_tseries = len(tseries_folders)
        if num_syncs != num_tseries:
            raise ValueError("Number of TSeries does not match Number of Syncs")

        # Get the number of TIF files per TSeries folder
        num_tifs = [len([f for f in os.listdir(os.path.join(self.raw_data_folder, folder)) if 'Ch2' in f]) for folder in tseries_folders]

        for acq_number in range(num_syncs):
            print(f"Imaging Dir: {tseries_folders[acq_number]}, Sync File: {sync_files[acq_number]}")
            sync_file_path = os.path.join(self.sync_base_path, sync_files[acq_number])
            imaging_folder = tseries_folders[acq_number]

            info = {
            'imaging_id': os.path.basename(tseries_folders[acq_number]),
            'sync_id': os.path.basename(sync_files[acq_number]),
            'frame_times': None,
            'frame_rate': None,
            'imaging_frame_rate': None,
            'sync_sampling_rate': None,
            'points': [],  # Initialize empty points list
            'point_order': [],  # Initialize empty point order list
            'microns_per_pixel': None,
            'galvo_signal_norm': []
            }

            # Load sync data based on file type
            if is_pclamp:
                reader = AxonIO(sync_file_path)
                segments = reader.read_block().segments[0]
                #if channel_number >= segments.analogsignals[0].shape[1]:
                #    raise IndexError(f"Channel number {channel_number} is out of bounds for available channels.")
                sync_data = segments.analogsignals[2][:, channel_number].magnitude.flatten()
                sync_sampling_rate = float(segments.analogsignals[0].sampling_rate)

            galvo_signal_norm = sync_data / np.max(sync_data)
            peaks, _ = find_peaks(galvo_signal_norm, height=0.3, prominence=0.1)
            scanning_amplitude = np.mean(sync_data[peaks])

            # if not passing in frame times from earlier alignment code, it will calculate the frame times
            if frame_times_list == None:
                good_peaks = [p for p in peaks if 0.95 * scanning_amplitude < sync_data[p] < 1.1 * scanning_amplitude]
                frame_times = np.array(good_peaks)
            else:
                frame_times = frame_times_list[acq_number]

            # Calculate frame rate and intervals
            frame_intervals_sec = np.diff(frame_times) / sync_sampling_rate
            imaging_frame_rate = 1 / (np.mean(frame_intervals_sec))
            # alignment_info.append({
            #     "imaging_id": imaging_folder,
            #     "sync_id": sync_files[acq_number],
            #     "sync_sampling_rate": sync_sampling_rate,
            #     "frame_times": frame_times
            # })

            # Optionally plot the galvo signal and frame detection
            # Adjust frame times if necessary based on TIF file count
            if len(frame_times) > num_tifs[acq_number]:
                frame_times = frame_times[:num_tifs[acq_number]]
    
            frame_intervals_sec = np.diff(frame_times) * (1/sync_sampling_rate)
            frame_rate = np.round(1/frame_intervals_sec)
            imaging_frame_rate = 1/(np.median(np.diff(frame_times))/sync_sampling_rate)
            info['frame_times'] = frame_times
            info['frame_rate'] = imaging_frame_rate
            info['imaging_frame_rate'] = imaging_frame_rate
            info['galvo_signal_norm'] = galvo_signal_norm
            info['sync_sampling_rate'] = sync_sampling_rate
        
        
            # Check if we have more frame times than expected
            frames_per_folder = self.get_frames_per_folder()
            if len(frame_times) > frames_per_folder[acq_number]:
                frame_times = frame_times[:int(frames_per_folder[acq_number])]
            
        
            alignment_info.append(info)

        self.alignment_info = alignment_info
        return alignment_info

    def plot_frame_times(self, num, galvo_signal, frame_times,plot_on):
        # First plot: Galvo signal and frame times
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        plt.clf()
        plt.plot(galvo_signal)
        plt.plot(frame_times, galvo_signal[frame_times], '*r')
        plt.title('Galvo Signal and Defined Frame Times')

        
        # Second plot: Histogram of frame intervals
        plt.subplot(2,1,2)
        plt.hist(np.diff(frame_times))
        plt.title('Unique frame intervals - if seeing range of values then check alignment carefully')
        plt.gca().tick_params(labelsize=12)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        galvo_plot_dir = os.path.join(self.save_folder, 'imaging_plots')
        os.makedirs(galvo_plot_dir, exist_ok=True)
        filename = f"spiral_detection_acq_{num:03d}.png"
        plt.savefig(os.path.join(galvo_plot_dir, filename))
        if plot_on:
            plt.show()
        else:
            plt.close()

      
        #plt.close()

    def get_spiral_locations(self, min_height=0.1):
        """Get spiral scanning locations and timing information using FFT-based detection."""
        # Initialize lists
        acquisition_data = []
        alignment_info = []
        
        # Get directories and files
        tseries_dirs = sorted(glob.glob(os.path.join(self.raw_data_folder, 'TSeries-*')))
        sync_files = sorted(glob.glob(os.path.join(self.sync_base_path, '*.abf')))
        
        # Process each acquisition
        for acq_number, (tseries_dir, sync_file) in enumerate(zip(tseries_dirs, sync_files)):
            # Initialize info dictionary
            info = {
                'imaging_id': os.path.basename(tseries_dir),
                'sync_id': os.path.basename(sync_file)
            }
            
            # Get the XML file path - looking for specific Cycle00001 file
            base_name = os.path.basename(tseries_dir)
            xml_file = os.path.join(tseries_dir, f"{base_name}_Cycle00001_MarkPoints.xml")
            
            base_name = os.path.basename(tseries_dir)
            meta_data_file = os.path.join(tseries_dir, f"{base_name}.xml")

            meta_data = self._parse_xml_metadata(meta_data_file)
            microns_per_pixel = float(meta_data['PVStateShard']['PVStateValue'][16]['IndexedValue'][0]['valueAttribute'])
            #print(microns_per_pixel)

            if not os.path.exists(xml_file):
                print(f"Warning: Missing XML file for acquisition {acq_number}")
                continue
                
            # Parse XML metadata
            mark_points = self._parse_xml_metadata(xml_file)
            
            # Extract spiral parameters from XML
            points = np.zeros((len(mark_points['PVMarkPointElement']), 2))
            point_order = np.zeros(len(mark_points['PVMarkPointElement'])) 
            

            # Get points from mark points XML
            for n, mark_point in enumerate(mark_points['PVMarkPointElement']):
                idx = int(mark_point['PVGalvoPointElement']['IndicesAttribute'])
                point_order[n] = idx - 1  # Convert to 0-based indexing
                points[idx-1, 0] = float(mark_point['PVGalvoPointElement']['Point']['XAttribute'])
                points[idx-1, 1] = float(mark_point['PVGalvoPointElement']['Point']['YAttribute'])
                
            # Trim points to unique ordered points
            unique_points = len(np.unique(point_order))
            points = points[:unique_points, :]
            
            # Scale points to image size
            image_size = np.array([self.ops['Lx'], self.ops['Ly']], dtype=float)
            points = points * image_size
            
            # Store point information
            info['points'] = points
            info['point_order'] = point_order
            info['iterations_per_acqs'] = mark_points['IterationsAttribute']
            
            # Calculate expected frequency from spiral parameters
            spiral_revolutions = float(mark_points['PVMarkPointElement'][0]['PVGalvoPointElement']['SpiralRevolutionsAttribute'])
            duration_ms = float(mark_points['PVMarkPointElement'][0]['PVGalvoPointElement']['DurationAttribute'])
            revolutions_per_ms = spiral_revolutions / duration_ms
            expected_freq = revolutions_per_ms * 1000  # Hz
            reps = float(mark_points['PVMarkPointElement'][0]['RepetitionsAttribute'])
            
            # Create bandpass filter around expected frequency
            sampling_rate = 10000  # Hz get from metadata
            freq_bandwidth = expected_freq * 0.5
            lowcut = max(0, expected_freq - freq_bandwidth)
            highcut = expected_freq + freq_bandwidth

            def butter_bandpass(lowcut, highcut, fs, order=5):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
                return b, a

            # Load and process sync data
            sync_data = self._load_sync_data(sync_file)
            galvo_x = sync_data[:, 6]  # Adjust indices as needed
            galvo_y = sync_data[:, 7]

            # Apply filter
            b, a = butter_bandpass(lowcut, highcut, sampling_rate)
            filtered_signal = filtfilt(b, a, galvo_x)

            # Set minimum distance to duration of one spiral
            min_distance = int(sampling_rate * duration_ms/1000) * reps # Convert ms to samples

            # Find peaks
            spiral_end_times = find_peaks(np.abs(filtered_signal), 
                                        height=min_height,
                                        distance=min_distance)[0]
            
             # Remove first and last peaks
            if len(spiral_end_times) > 2:  # Only remove if we have at least 3 peaks
                spiral_end_times = spiral_end_times[1:-1]
            else:
                print(f"Warning: Acquisition {acq_number} has fewer than 3 peaks detected")
                spiral_end_times = np.array([])  # Empty array if not enough peaks

            # Calculate start times
            spiral_duration = int(duration_ms * sampling_rate/1000)  # Convert ms to samples
            spiral_start_times = spiral_end_times - min_distance
            spiral_start_times = np.array(spiral_start_times, dtype=int)
            # Store timing information
            info['num_spirals'] = len(spiral_end_times)
            info['spiral_end_times'] = spiral_end_times
            info['spiral_start_times'] = spiral_start_times
            info['spiral_duration_ms'] = duration_ms
            info['microns_per_pixel'] = microns_per_pixel
            info['galvo_x_start'] = galvo_x[spiral_start_times]
            info['galvo_y_start'] = galvo_y[spiral_start_times]
            info['galvo_x_end'] = galvo_x[spiral_end_times]
            info['galvo_y_end'] = galvo_y[spiral_end_times]

            # Store in acq_data
            acq_data = {
                'galvo_x': galvo_x,
                'filtered_signal': filtered_signal,
                'spiral_end_times': spiral_end_times,
                'spiral_start_times': spiral_start_times,
                'acq_number': acq_number,
                'points': points.copy(),
                'point_order': point_order.copy(),
                'duration_ms': duration_ms,
                'reps': reps,
                'microns_per_pixel': microns_per_pixel
            }
            acquisition_data.append(acq_data)
            
            # Update alignment_info with the modified info dictionary
            self.alignment_info[acq_number].update(info)
        
        return acquisition_data

    def process_targets_and_trials(self, pre_trial_window=30, post_trial_window=30):
        """
        Main processing function to find targets and analyze trials
        
        Args:
            pre_trial_window (int): Number of frames before stimulation to include in trial window (default: 30)
            post_trial_window (int): Number of frames after stimulation to include in trial window (default: 30)
        """
        # Create save directory for ROI plots
        os.makedirs(os.path.join(self.save_folder, 'roi plots'), exist_ok=True)

        # Calculate offset information
        microns_per_pixel = float(self.alignment_info[0]['microns_per_pixel'])
        
        xoff_um, yoff_um, distoff_um = self._get_offset_info(
            self.ops['xoff'], 
            self.ops['yoff'], 
            microns_per_pixel
        )
        
        # Save offset info
        np.savez(os.path.join(self.base, 'offsetInfo.npz'),
                 xoff=self.ops['xoff'],
                 yoff=self.ops['yoff'],
                 xoff_um=xoff_um,
                 yoff_um=yoff_um,
                 distoff_um=distoff_um)

        # Get consolidated points across acquisitions
        # Only neccessary if there are a different number of points or different points in each acquisition
        # Looks at the unique points in each acquisition and matches those with locations of each  
        ################################################ 
        temp_points = []
        temp_point_order = []
        for acq_info in self.alignment_info:
            temp_points.append(acq_info['points'])
            temp_point_order.append(np.unique(acq_info['point_order']))

        temp_points = np.vstack(temp_points)
        temp_point_order = np.concatenate(temp_point_order)
        
        # Get unique points and their final positions
        point_order = np.unique(temp_point_order)
        points = np.zeros((len(point_order), 2))
        for p, po in enumerate(point_order):
            idx = np.where(temp_point_order == po)[0][-1]
            points[p] = temp_points[idx]

        ################################################

        # Find target ROIs
        target_id, dist_id = self._find_targets(
            points, 
            self.stat, 
            microns_per_pixel,  # Now passing as float
            self.ops
        )


        # Create trial frames
        framerate = self.alignment_info[0]['imaging_frame_rate']
        frames_per_folder = self.get_frames_per_folder()
        block_edges = np.cumsum(frames_per_folder)
        
        # Initialize stim_id array
        total_frames = int(np.sum(frames_per_folder))
        stim_id = np.zeros(total_frames)
        
        # Process stimulation times for each acquisition
        current_start_frame = 0
        for acq_num, acq_info in enumerate(self.alignment_info):
            if len(acq_info['points']) > 0:
                block_frames = np.arange(
                    block_edges[acq_num] - frames_per_folder[acq_num],
                    block_edges[acq_num]
                )
                                
                for stim_idx, stim_time in enumerate(acq_info['spiral_start_times']):
                    # Find closest frame to stim time
                    frame_idx = np.argmin(np.abs(
                        acq_info['frame_times'] - stim_time
                    ))
                    
                    # Get point order index
                    # point_idx = stim_idx % len(acq_info['point_order'])
                    # if point_idx == 0:
                    #     point_idx = len(acq_info['point_order'])


                    # TO DO: 
                    # CHANGE - assign point index based on the position of  the modded stim_idx in the point order array 
                    # SPECIFIC to each aquisition
                    

                    point_idx = int(stim_idx % len(acq_info['point_order']))

                    # Record stimulation
                    abs_frame = int(frame_idx + current_start_frame)

                    if not (abs_frame > total_frames):
                        stim_id[abs_frame] = acq_info['point_order'][point_idx] + 1
                
                current_start_frame = block_edges[acq_num]

        # Create trial windows (removed hardcoded values)
        trial_frames, full_trials = self._create_trial_windows(
            stim_id, points, pre_trial_window, post_trial_window, block_edges
        )


        trial_info = {
            'trial_frames': trial_frames,
            'full_trials': full_trials,
            'target_id': target_id,
            'dist_id': dist_id,
            'stim_id': stim_id
        }
            

        self.trial_info = trial_info


        # Save results with window information
        self._save_results(
            trial_frames=trial_frames,
            block_edges=block_edges,
            framerate=framerate,
            frames_per_folder=frames_per_folder,
            points=points,
            stim_id=stim_id,
            full_trials=full_trials,
            pre_trial_window=pre_trial_window,  # Save window sizes
            post_trial_window=post_trial_window
        )

    def _get_offset_info(self, xoff, yoff, microns_per_pixel):
        """Calculate offset information in microns"""
        microns_per_pixel = float(microns_per_pixel)

        xoff = np.array(xoff, dtype=float)
        yoff = np.array(yoff, dtype=float)

        xoff_um = xoff * microns_per_pixel
        yoff_um = yoff * microns_per_pixel
        distoff_um = np.sqrt(xoff_um**2 + yoff_um**2)
        return xoff_um, yoff_um, distoff_um

    def _create_trial_windows(self, stim_id, points, pre_window, post_window, block_edges):
        """Create trial windows around stimulation times"""
        
        trial_frames = []
        full_trials = []
        
        for point_idx in range(len(points)):
            stim_times = np.where(stim_id == point_idx + 1)[0]
            
            # Initialize trial frames and validity
            point_trials = np.zeros((len(stim_times), pre_window + post_window + 1))
            trial_valid = np.zeros(len(stim_times))
            
            for trial_idx, stim_time in enumerate(stim_times):
                if stim_time > pre_window:
                    window = np.arange(
                        stim_time - pre_window,
                        stim_time + post_window + 1
                    )
                    point_trials[trial_idx] = window
                    trial_valid[trial_idx] = 1
                    
                    # Check if trial crosses block boundaries
                    block = np.searchsorted(block_edges, stim_time)
                    if not self._check_trial_in_block(window, block, block_edges):
                        trial_valid[trial_idx] = 0
            
            trial_frames.append(point_trials)
            full_trials.append(trial_valid)
            
        return trial_frames, full_trials

    def _check_trial_in_block(self, trial_frames, block, block_edges):
        """Check if trial frames are within the same acquisition block"""
        if block == 0:
            return trial_frames[-1] < block_edges[0]
        return (trial_frames[0] >= block_edges[block-1] and 
                trial_frames[-1] < block_edges[block])

    def _save_results(self, **kwargs):
        """Save analysis results with proper handling of inhomogeneous arrays"""
        save_path = os.path.join(self.base, 'trial_and_target_info.npz')
        
        # Create a new dict with properly formatted arrays
        save_dict = {}
        
        for key, val in kwargs.items():
            if isinstance(val, (list, np.ndarray)):
                try:
                    # Try to convert to regular numpy array
                    save_dict[key] = np.asarray(val)
                except ValueError:
                    # If that fails, use object array
                    save_dict[key] = np.array(val, dtype=object)
            else:
                # Non-array data can be saved directly
                save_dict[key] = val
        
        # Save using the processed dictionary
        np.savez(save_path, **save_dict)

    def _find_targets(self, points, stat, micronsperpix, ops):
        """
        For each spiral finds the target cell and the cells in the immediate area of the spirals.
        
        Args:
            points (np.ndarray): pixel coordinates for center of photostim
            stat (list): suite2p structure with cell location information
            micronsperpix (float): microns per pixel conversion factor
            ops (dict): suite2p ops dictionary containing max projection
            
        Returns:
            tuple: (targetID, distID) lists containing cell IDs and distances for each point
        """
        # Ensure micronsperpix is float
        micronsperpix = float(micronsperpix)
        
        # Get median coordinates for all cells
        median_all = np.zeros((len(stat), 2))
        for n in range(len(stat)):
            median_all[n, :] = np.flip(np.array(stat[n]['med']).astype(float))
        
        # Use all cells as potential targets
        cell_ind = np.arange(len(stat))
        
        # Initialize output lists
        target_id = []
        dist_id = []
        
        # Find location of target cell for each point
        for point in points:
            # Calculate distances to all cells
            dist = np.sqrt(np.sum((median_all[cell_ind] - point)**2, axis=1))
            dist = dist * micronsperpix  # Now micronsperpix is guaranteed to be float
            
            # Find cells within 20 microns
            min_ind = np.where(dist < 20)[0]
            target_id.append(cell_ind[min_ind])
            dist_id.append(dist[min_ind])
        
        # Create figure showing ROIs        
        self.plot_ROI_locations(target_id,points,1)
        # for i in range(len(points)):
        #     plt.clf()
        #     plt.imshow(img)
        #     plt.scatter(points[i, 0], points[i, 1], s=100, c='b', marker='o')
            
        #     # Plot ROI boundaries for each target
        #     for n in range(len(target_id[i])):
        #         try:
        #             self._plot_mask_boundaries(target_id[i][n], stat, color='r')
        #         except:
        #             continue
                    
        #     plt.savefig(os.path.join(self.save_folder, f'{i+1} roi check.png'))
        #     plt.close()
        
        return target_id, dist_id

    def _plot_mask_boundaries(self, cell_idx, stat, color='r'):
        """
        Helper function to plot ROI boundaries.
        
        Args:
            cell_idx (int): Index of cell to plot
            stat (list): suite2p stat structure
            color (str): Color of boundary line
        """
        ypix = stat[cell_idx]['ypix']
        xpix = stat[cell_idx]['xpix']
        
        # Find boundary pixels
        boundary_mask = np.zeros((np.ptp(ypix) + 4, np.ptp(xpix) + 4))
        boundary_mask[
            ypix - np.min(ypix) + 2,
            xpix - np.min(xpix) + 2
        ] = 1
        
        # Find contours
        contours = measure.find_contours(boundary_mask, 0.5)
        
        # Plot the largest contour
        if len(contours) > 0:
            largest_contour = max(contours, key=len)
            y_plot = largest_contour[:, 0] + np.min(ypix) - 2
            x_plot = largest_contour[:, 1] + np.min(xpix) - 2
            plt.plot(x_plot, y_plot, color, linewidth=1)

    def _parse_xml_metadata(self, xml_file):
        """
        Parse XML metadata file into a dictionary structure.
        
        Args:
            xml_file (str): Path to XML metadata file

        Returns:
            dict: Parsed metadata with nested structure
        """
        def element_to_dict(element):
            result = {}
            
            # Add attributes if present
            if element.attrib:
                for key, value in element.attrib.items():
                    result[key + 'Attribute'] = value
            
            # Process child elements
            for child in element:
                child_data = element_to_dict(child)
                
                # Handle multiple children with same tag
                if child.tag in result:
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = child_data
                    
            # Add text content if present and no children
            if element.text and element.text.strip():
                result['text'] = element.text.strip()
                
            return result

        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Convert to dictionary
        metadata = element_to_dict(root)
        
        # Debug print to check structure
        print(f"\nParsing {os.path.basename(xml_file)}")
        print("Top level keys:", list(metadata.keys()))
        
        return metadata
    
    def _load_sync_data(self, sync_file):
        """
        Load sync data from abf file.

        Args:
            sync_file (str): Path to .abf file
                
        Returns:
            np.ndarray: Array containing sync data with shape (samples, channels)
        """
        # Load ABF file
        abf = pyabf.ABF(sync_file)
        
        # Get number of channels
        num_channels = abf.channelCount
        
        # Create array to store all channel data
        sync_data = np.zeros((len(abf.sweepY), num_channels))
    
        # Load data from each channel
        for ch in range(num_channels):
            abf.setSweep(0, channel=ch)
            sync_data[:, ch] = abf.sweepY
            
        return sync_data
    
    def _filter_galvo_sig(self, signal, sample_rate, cutoff_freq):
        """
        Apply low-pass filter to galvo signal.
        
        Args:
            signal (np.ndarray): Input signal
            sample_rate (float): Sampling rate in Hz
            cutoff_freq (float): Cutoff frequency in Hz
            
        Returns:
            np.ndarray: Filtered signal
        """
        from scipy import signal as sig
        
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = sig.butter(2, normalized_cutoff, btype='low')
        filtered = sig.filtfilt(b, a, signal)
        
        return filtered
    
    def _smooth(self, input_data, window, window_type='flat'):
        """
        Smooth the data using a window with requested size and type.
        
        Args:
            input_data (np.ndarray): Input signal
            window (int): Smoothing window size
            window_type (str): Window type ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')
        
        Returns:
            np.ndarray: Smoothed signal
        """
        # Handle input dimensions
        if len(input_data) < window:
            raise ValueError("Input vector needs to be bigger than window size.")
        
        # Process in chunks if data is too large
        chunk_size = 1000000  # Adjust this value based on your memory constraints
        if len(input_data) > chunk_size:
            # Process in chunks
            num_chunks = int(np.ceil(len(input_data) / chunk_size))
            smoothed_data = np.zeros_like(input_data)
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(input_data))
                
                # Process chunk with overlap to avoid edge effects
                overlap = window * 2
                chunk_start = max(0, start_idx - overlap)
                chunk_end = min(len(input_data), end_idx + overlap)
                
                # Smooth chunk
                chunk = input_data[chunk_start:chunk_end]
                smoothed_chunk = self._smooth_chunk(chunk, window, window_type)
                
                # Remove overlap and store result
                if i == 0:
                    smoothed_data[start_idx:end_idx] = smoothed_chunk[:end_idx-start_idx]
                elif i == num_chunks-1:
                    smoothed_data[start_idx:end_idx] = smoothed_chunk[overlap:]
                else:
                    smoothed_data[start_idx:end_idx] = smoothed_chunk[overlap:-overlap]
                    
            return smoothed_data
        else:
            return self._smooth_chunk(input_data, window, window_type)

    def _smooth_chunk(self, input_data, window, window_type='flat'):
        """Helper function to smooth a single chunk of data"""
        if window_type not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window type must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        halfwin = window // 2
        input_length = len(input_data)
        
        # Create window
        if window_type == 'flat':
            kernel = np.ones(window)
        else:
            kernel = getattr(np, window_type)(window)
        kernel = kernel / kernel.sum()
        
        # Create padded array
        padded = np.pad(input_data, halfwin, mode='edge')
        
        # Convolve
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed
    
    def plot_spiral_detection(self, acq_data, save_dir=None, show_plot=True):
        """
        Plot spiral detection results for one acquisition.
        
        Args:
            acq_data (dict): Dictionary containing acquisition data
            save_dir (str, optional): Directory to save plots. If None, plots will be shown
            show_plot (bool): Whether to display the plot (default True)
        """
        # Create save directory if specified
        if save_dir:
            spiral_plot_dir = os.path.join(save_dir, 'spiral_detection_plots')
            os.makedirs(spiral_plot_dir, exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(15,10))
        
        # First subplot
        plt.subplot(211)
        plt.plot(acq_data['galvo_x'], 'b', label='Original')
        plt.plot(acq_data['filtered_signal'], 'g', label='Filtered')
        plt.scatter(acq_data['spiral_end_times'], 
                    acq_data['galvo_x'][acq_data['spiral_end_times']], 
                    c='r', label='End')
        plt.scatter(acq_data['spiral_start_times'], 
                    acq_data['galvo_x'][acq_data['spiral_start_times']], 
                    c='b', label='Start')
        plt.title(f"Acquisition {acq_data['acq_number']}")
        plt.legend()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Second subplot
        plt.subplot(212)
        plt.plot(acq_data['filtered_signal'])
        plt.scatter(acq_data['spiral_start_times'], 
                    acq_data['filtered_signal'][acq_data['spiral_start_times']], 
                    c='b')
        plt.title('Filtered Signal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        # Save or show based on parameters
        if save_dir:
            filename = f"spiral_detection_acq_{acq_data['acq_number']:03d}.png"
            plt.savefig(os.path.join(spiral_plot_dir, filename))
            plt.close()
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    def create_ROI_traces(self):  
        roi_plot_dir = os.path.join(self.save_folder, 'roi plots')
        os.makedirs(roi_plot_dir, exist_ok=True)

        # Process each ROI
        F_struct = {}    # Main fluorescence traces
        # F_struct2 = {}   # Neuropil-corrected fluorescence
        # Fneu_struct = {} # Neuropil traces
        roi_inds = {}    # Indices of ROIs in trial_info
        # Initialize structures for each target
        for target_id in range(len(self.alignment_info[-1]['points'])):
            F_struct[target_id] = {}
            roi_inds[target_id] = {}
            
            # Initialize arrays for each ROI in this target
            for roi_idx in self.trial_info['target_id'][target_id]:
                F_struct[target_id][roi_idx] = []
                roi_inds[target_id][roi_idx] = []

        # Fill structures with trial data
        for target_id in range(len(self.alignment_info[-1]['points'])):
            for roi_idx in self.trial_info['target_id'][target_id]:
                for tr in range(len(self.trial_info['trial_frames'][target_id])):
                    if np.max(self.trial_info['trial_frames'][target_id][tr]) < self.F.shape[1]:
                        frames = self.trial_info['trial_frames'][target_id][tr].astype(int)
                        
                        # Store different versions of fluorescence traces
                        F_trace = self.F[roi_idx, frames]
                        Fneu_trace = self.Fneu[roi_idx, frames]
                        F_corrected = F_trace - 0.7 * Fneu_trace
                        F_struct[target_id][roi_idx].append(F_corrected)
                        roi_inds[target_id][roi_idx] = roi_idx
        return F_struct, roi_inds

    def plot_ROI_traces(self, F_struct, pre_trial_window):
        for i in range(len(self.alignment_info[-1]['points'])):
            
            
            # Get ROIs for this target
            target_rois = self.trial_info['target_id'][i]
            dist_ids = np.array(self.trial_info['dist_id'][i])
            order = np.argsort(dist_ids)

            length = 5*(len(target_rois[order]))
            plt.figure(figsize=(length, 5))
            
            for n, roi_idx in enumerate(target_rois[order]):
                
                plt.subplot(2, len(target_rois), n+1)
                
                # Plot all trials and mean if we have data for this ROI
                if roi_idx in F_struct[i]:
                    # Get list of traces
                    traces_list = F_struct[i][roi_idx]
                    
                    # Ensure all traces have the same length
                    min_length = min(len(trace) for trace in traces_list)
                    traces_array = np.array([trace[:min_length] for trace in traces_list])
                    
                    # Calculate the mean across trials
                    mean_trace = np.mean(traces_array, axis=0)
                    
                    # Plot each trial
                    # for trace in traces_array:
                    #     plt.plot(trace.T, 'gray', alpha=0.3)  # Simply add .T here
                    
                    # Plot the mean trace
                    plt.plot(mean_trace, 'k', linewidth=2)
                    plt.axvline(x=pre_trial_window,color='r',linestyle='--',linewidth=3)
                    plt.title(f"ROI {roi_idx}")
                    ax = plt.gca()
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)

            
            for n, roi_idx in enumerate(target_rois[order]):
                plt.subplot(2,len(target_rois),(n+1)+len(target_rois))
                ops = self.ops
                stat = self.stat 
                img = (ops['max_proj'] - ops['max_proj'].min()) / (ops['max_proj'].max() - ops['max_proj'].min())
                plt.scatter(self.alignment_info[-1]['points'][i, 0], self.alignment_info[-1]['points'][i, 1], s=50, c='k', marker='o')
                plt.imshow(img)
                plt.title('ROI ' + str(roi_idx))
                plt.axis('off')
                self._plot_mask_boundaries(self.trial_info['target_id'][i][n], stat, color='r')

                

            # plt.subplot(2,len(target_rois), len(target_rois[order])+1)
            # for ii in range(len(self.trial_info['target_id'][i])):
            #     try:
            #         self._plot_mask_boundaries(self.trial_info['target_id'][i][ii], stat, color='r')
            #     except:
            #         continue

            roi_plot_dir = os.path.join(self.save_folder, 'roi plots')
            #plt.tight_layout()
            plt.savefig(os.path.join(roi_plot_dir, f'{i+1} candidate rois neu correct.png'))
            plt.show()

    def plot_ROI_locations(self,target_id,points,saveON):
        # Create figure showing ROIs
        ops = self.ops
        stat = self.stat 
        img = (ops['max_proj'] - ops['max_proj'].min()) / (ops['max_proj'].max() - ops['max_proj'].min())
        
        for i in range(len(points)):
            plt.figure()
            plt.imshow(img)
            plt.scatter(points[i, 0], points[i, 1], s=100, c='b', marker='o')
            
            # Plot ROI boundaries for each target
            for n in range(len(target_id[i])):
                try:
                    self._plot_mask_boundaries(target_id[i][n], stat, color='r')
                except:
                    continue
            
            if saveON:
                plt.savefig(os.path.join(self.save_folder, f'{i+1} roi check.png'))
                plt.close()
            
    def find_s2p_and_dff_indices(self,celi,target_cells):
        suite2p_folder_red = self.suite2p_folder + r'\\Fall_red.mat'
        Fall_red = scipy.io.loadmat(suite2p_folder_red)

        suite2p_folder_all = self.suite2p_folder + r'\\Fall.mat'
        Fall= scipy.io.loadmat(suite2p_folder_all)

        red_cells = np.where(Fall_red['iscell'][:,0]==1)[0]
        all_cells = np.where(Fall['iscell'][:,0]==1)[0]
        non_cells = np.setdiff1d(all_cells,red_cells)

        # Save non-cell suite2p inds
        s2p_indices = {
            'non_cell_inds' : non_cells,
            'red_cell_inds' : red_cells,
            'target_cell_inds' : target_cells
        }

        non_cell_dff = np.where(np.isin(celi,non_cells))[0]
        red_cell_dff = np.where(np.isin(celi,red_cells))[0]
        target_cell_dff = np.where(np.isin(celi,target_cells))[0]

        dff_inds = {
            's2p_dff_inds' : celi,
            'non_cell_dff' : non_cell_dff,
            'red_cell_dff' : red_cell_dff,
            'target_cell_dff' : target_cell_dff
        }

        indices = {
            's2p_indices' : s2p_indices,
            'dff_inds' : dff_inds
        }


        save_path = self.base
        hdf5storage.savemat(os.path.join(save_path,'indices.mat'),indices, format='7.3', oned_as='column',truncate_existing=True)
        # Save the list of imaged trials

        with open(f'{save_path}/indices.pkl', 'wb') as f:
            pickle.dump(indices, f)

    def add_stim_id_to_image_spk(self):
        p= os.path.join(self.base, 'imaging_data.pkl')
        stim_id = self.trial_info['stim_id']
        with open(p, "rb") as file:
            imaging_dict = pickle.load(file)
        for trial in imaging_dict:
            if imaging_dict[trial]['good_trial'] == 1: 
                ITI = imaging_dict[trial]['movement_in_imaging_time']['in_ITI']
                frames = imaging_dict[trial]['relative_frames']

                # TEMP ; FIX IMAGE STRUCT CODE  
                frames = frames[0:len(frames)-1]
                # TEMP ; FIX IMAGE STRUCT CODE  
                
                section = stim_id[frames]
                imaging_dict[trial]["stim_id_trial"] = section
            else:
                imaging_dict[trial]["stim_id_trial"] = None
                        
        with open(p,"wb") as file:
            pickle.dump(imaging_dict,file)


        imaging_sorted = {}
        prefix = 'Trial_'

        # Add zero buffers to trial numbers of 1,2 digits (helps with matlab sorting)
        for k in imaging_dict:
            dict_key = f"{prefix}{int(k.split('_')[1]):03}"
            imaging_sorted[dict_key] = imaging_dict[k]


        # OVERWRITE .MAT VERSION 
        imaging_dict = {
            'imaging_dict' : imaging_sorted
        }
        hdf5storage.savemat(os.path.join(self.base,'imaging_py.mat'),imaging_dict, format='7.3', oned_as='column',truncate_existing=True)

