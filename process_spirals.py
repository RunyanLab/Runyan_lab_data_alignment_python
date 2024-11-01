import numpy as np
import os
import glob
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import suite2p as s2p
from suite2p.io import load_suite2p
import h5py
import pyabf
from skimage import measure
import xml.etree.ElementTree as ET
from scipy import signal as sig

class GalvoTargetAnalysis:
    def __init__(self):
        # Initialize paths
        self.raw_data_folder = None
        self.suite2p_folder = None
        self.sync_base_path = None
        self.save_folder = None
        
        # Data containers
        self.stat = None
        self.ops = None
        self.F = None
        self.Fneu = None
        self.spks = None
        self.s2p_badframes = None
        self.nchannels = None  # Added to store number of channels
        self.alignment_info = None  # Store alignment info
        
    def get_directories(self):
        """Get all necessary directories using dialog boxes"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        self.raw_data_folder = filedialog.askdirectory(title='Pick Directory Containing 2P Acquisitions')
        self.suite2p_folder = filedialog.askdirectory(title='Pick Directory Containing Suite2P Files')
        self.sync_base_path = filedialog.askdirectory(title='Pick Directory Containing Sync Files')
        self.save_folder = filedialog.askdirectory(title='Pick Directory to Save These Analysis Files')
        
    def load_suite2p_data(self):
        """Load Suite2p data"""
        # Load Suite2p output
        s2p_data = load_suite2p(self.suite2p_folder)
        
        self.stat = s2p_data['stat']
        self.ops = s2p_data['ops']
        self.F = s2p_data['F']
        self.Fneu = s2p_data['Fneu']
        self.spks = s2p_data['spks']
        
        # Get bad frames
        self.s2p_badframes = np.where(self.ops['badframes'] > 0)[0]
        
    def get_frames_per_folder(self):
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
        
    def get_frame_times(self, channel_number=5, plot_on=True):
        """
        Get frame timing information from sync files and imaging data.
        
        Args:
            channel_number (int): The sync channel where slow galvo info is recorded
            plot_on (bool): Whether to show verification plots
        """
        # Check if paths are set
        if any(path is None for path in [self.raw_data_folder, self.sync_base_path]):
            raise ValueError("Paths not set. Run get_directories() first.")
            
        # Check for sync files (.abf or .h5)
        abf_files = sorted(glob.glob(os.path.join(self.sync_base_path, '*.abf')))
        h5_files = sorted(glob.glob(os.path.join(self.sync_base_path, '*.h5')))
        
        if len(abf_files) > 0:
            sync_files = abf_files
            is_pclamp = True
        else:
            sync_files = h5_files
            is_pclamp = False
        
        # Get TSeries directories
        tseries_dirs = sorted(glob.glob(os.path.join(self.raw_data_folder, 'TSeries-*')))
        
        # Verify matching numbers
        if len(sync_files) != len(tseries_dirs):
            raise ValueError(f'Number of TSeries ({len(tseries_dirs)}) does not match Number of Syncs ({len(sync_files)})')
        
        num_acqs = len(sync_files)
        
        # Get number of TIFs per series
        num_tifs = []
        for tseries_dir in tseries_dirs:
            ch2_files = glob.glob(os.path.join(tseries_dir, '*Ch2*'))
            num_tifs.append(len(ch2_files))
        
        # Print file matching for verification
        for acq_number in range(num_acqs):
            if acq_number == 0:
                print('CHECK THAT FILE NUMBER FROM VELOCITY AND TSERIES MATCHES UP')
            
            print(f"TSeries: {os.path.basename(tseries_dirs[acq_number])}")
            print(f"Sync: {os.path.basename(sync_files[acq_number])}")
            input("Press Enter to continue...")
        
        # Initialize alignment info list
        alignment_info = []
        
        # Process each acquisition
        for acq_number in range(num_acqs):
            info = {
                'imaging_id': os.path.basename(tseries_dirs[acq_number]),
                'sync_id': os.path.basename(sync_files[acq_number])
            }
            
            # Load sync data
            if is_pclamp:
                abf = pyabf.ABF(sync_files[acq_number])
                sync_sampling_rate = abf.dataRate
                abf.setSweep(0, channel=channel_number)
                galvo_signal = abf.sweepY
                info['sync_sampling_rate'] = sync_sampling_rate
            else:
                with h5py.File(sync_files[acq_number], 'r') as f:
                    # Implement h5 file reading based on your specific file structure
                    pass
            
            # Normalize galvo signal
            galvo_signal_norm = galvo_signal / np.max(galvo_signal)
            
            # Find frame times
            _, frame_times = find_peaks(galvo_signal_norm, 
                                      height=0.3, 
                                      prominence=0.1)
            
            # Calculate frame rate
            frame_intervals_sec = np.diff(frame_times) * (1/sync_sampling_rate)
            frame_rate = np.round(1/frame_intervals_sec)
            imaging_frame_rate = 1/(np.median(np.diff(frame_times))/sync_sampling_rate)
            
            # Plotting for verification
            if plot_on:
                plt.figure(87)
                plt.clf()
                plt.plot(galvo_signal)
                plt.plot(frame_times, galvo_signal[frame_times], '*r')
                plt.title('Galvo Signal and Defined Frame Times')
                plt.suptitle(f'Acq number: {acq_number}')
                plt.gca().set_fontsize(12)
                
                plt.figure(88)
                plt.clf()
                plt.hist(np.diff(frame_times))
                plt.title('Unique frame intervals - if seeing range of values then check alignment carefully')
                plt.suptitle(f'Acq number: {acq_number}')
                plt.gca().set_fontsize(12)
                plt.show()
                input("Press Enter to continue...")
            
            # Adjust frame times if needed
            if len(frame_times) > num_tifs[acq_number]:
                frame_times = frame_times[:num_tifs[acq_number]]
            
            # Calculate alternative frame times
            start_time = frame_times[0]
            frame_intervals = np.zeros(num_tifs[acq_number])
            frame_intervals[1:] = 1/imaging_frame_rate * sync_sampling_rate
            frame_intervals[0] = start_time
            alt_frame_times = np.cumsum(frame_intervals)
            
            # Additional plotting
            if plot_on and len(frame_times) == len(alt_frame_times):
                plt.figure(888)
                plt.clf()
                plt.plot(frame_times, alt_frame_times, 'o')
                plt.xlabel('Frame times defined by galvo signal')
                plt.ylabel('Frame times defined by first galvo frame and then by frame rate')
                plt.gca().set_fontsize(12)
                plt.axis('square')
                plt.show()
            elif plot_on:
                plt.figure(888)
                plt.clf()
                plt.plot(galvo_signal)
                plt.plot(alt_frame_times, galvo_signal[alt_frame_times.astype(int)], '*r')
                plt.show()
            
            input("Press Enter to continue...")
            
            # Store frame times
            if len(np.unique(frame_rate)) > 1:
                info['frame_times'] = alt_frame_times
            else:
                info['frame_times'] = frame_times
                
            info['frames_times'] = frame_times
            alignment_info.append(info)
        
        self.alignment_info = alignment_info
        
    def get_spiral_locations(self, min_height=0.5, min_prom=0.1):
        """
        Get spiral scanning locations and timing information.
        
        Args:
            min_height (float): Minimum peak height for spiral detection
            min_prom (float): Minimum peak prominence for spiral detection
        """
        # Check if paths are set
        if any(path is None for path in [self.raw_data_folder, self.sync_base_path]):
            raise ValueError("Paths not set. Run get_directories() first.")
        
        # Load metadata from XML files
        xml_files = sorted(glob.glob(os.path.join(self.raw_data_folder, '**/*.xml'), recursive=True))
        if not xml_files:
            raise ValueError("No XML metadata files found")
        
        # Parse metadata files
        mark_points = []
        meta_data = []
        for i in range(0, len(xml_files), 2):
            # Assuming alternating pattern of env and mark files like in MATLAB
            mark_file = xml_files[i+1]
            env_file = xml_files[i]
            
            # Use XML parsing library of choice (e.g., ElementTree)
            mark_points.append(self._parse_xml_metadata(mark_file))
            meta_data.append(self._parse_xml_metadata(env_file))
        
        # Find acquisition directories
        tseries_dirs = sorted(glob.glob(os.path.join(self.raw_data_folder, 'TSeries-*')))
        if not tseries_dirs:
            tseries_dirs = sorted(glob.glob(os.path.join(self.raw_data_folder, 'MultipleTargets-*')))
        
        # Find sync files
        sync_files = sorted(glob.glob(os.path.join(self.sync_base_path, '*.h5')))
        
        # Verify matching numbers
        if len(sync_files) != len(tseries_dirs):
            raise ValueError(f'Number of TSeries ({len(tseries_dirs)}) does not match Number of Syncs ({len(sync_files)})')
        
        num_acqs = len(sync_files)
        alignment_info = []
        
        # Process each acquisition
        for acq_number in range(num_acqs):
            info = {
                'imaging_id': os.path.basename(tseries_dirs[acq_number]),
                'sync_id': os.path.basename(sync_files[acq_number])
            }
            
            # Get point coordinates from metadata
            points = np.zeros((len(mark_points[acq_number]['PVMarkPointElement']), 2))
            point_order = np.zeros(len(mark_points[acq_number]['PVMarkPointElement']))
            
            for n, mark_point in enumerate(mark_points[acq_number]['PVMarkPointElement']):
                idx = mark_point['PVGalvoPointElement']['IndicesAttribute']
                point_order[n] = idx
                points[idx, 0] = mark_point['PVGalvoPointElement']['Point']['XAttribute']
                points[idx, 1] = mark_point['PVGalvoPointElement']['Point']['YAttribute']
            
            # Trim points to unique ordered points
            unique_points = len(np.unique(point_order))
            points = points[:unique_points, :]
            
            # Scale points to image size
            image_size = np.array([self.ops['Lx'], self.ops['Ly']], dtype=float)
            points = points * image_size
            
            # Store point information
            info['points'] = points
            info['point_order'] = point_order
            info['iterations_per_acqs'] = mark_points[acq_number]['IterationsAttribute']
            info['microns_per_pixel'] = meta_data[acq_number]['PVStateShard']['PVStateValue'][15]['IndexedValue']['valueAttribute']
            
            # Calculate spiral duration
            spiral_duration = (mark_points[acq_number]['PVMarkPointElement'][0]['PVGalvoPointElement']['DurationAttribute'] * 
                             mark_points[acq_number]['PVMarkPointElement'][0]['RepetitionsAttribute'])
            info['spiral_duration_ms'] = spiral_duration
            
            # Load and process sync data
            with h5py.File(sync_files[acq_number], 'r') as f:
                sync_data = self._load_sync_data(f)  # Implement based on your h5 structure
                galvo_x = sync_data[:, 1]  # Adjust indices as needed
                galvo_y = sync_data[:, 2]
                
            # Filter galvo signals
            filtered_x = self._filter_galvo_sig(galvo_x, 1000, 2)
            filtered_y = self._filter_galvo_sig(galvo_y, 1000, 2)
            filtered_galvos = filtered_x + filtered_y
            
            # Find spiral times
            temp = np.abs(filtered_galvos)
            temp = self._smooth(temp, 25)
            
            # Find peaks for spiral end times
            peaks, spiral_end_times = find_peaks(temp, height=min_height, prominence=min_prom)
            spiral_end_times = spiral_end_times[1:]  # Skip first (junk) peak
            spiral_start_times = spiral_end_times - spiral_duration
            
            # Store timing information
            info['num_spirals'] = len(spiral_end_times)
            info['spiral_end_times'] = spiral_end_times
            info['spiral_start_times'] = spiral_start_times
            
            # Optional visualization
            plt.figure()
            plt.plot(galvo_x)
            plt.scatter(spiral_end_times, galvo_x[spiral_end_times], c='r')
            plt.scatter(spiral_start_times, galvo_x[spiral_start_times], c='b')
            plt.title(f'Acquisition {acq_number}')
            plt.show()
            
            alignment_info.append(info)
        
        self.alignment_info = alignment_info

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
        xoff_um, yoff_um, distoff_um = self._get_offset_info(
            self.ops['xoff'], 
            self.ops['yoff'], 
            self.alignment_info[-1]['microns_per_pixel']
        )
        
        # Save offset info
        np.savez(os.path.join(self.save_folder, 'offsetInfo.npz'),
                 xoff=self.ops['xoff'],
                 yoff=self.ops['yoff'],
                 xoff_um=xoff_um,
                 yoff_um=yoff_um,
                 distoff_um=distoff_um)

        # Get consolidated points across acquisitions
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

        # Find target ROIs
        target_id, dist_id = self._find_targets(points, self.stat, 
                                              self.alignment_info[-1]['microns_per_pixel'],
                                              self.ops)

        # Create trial frames
        framerate = self.alignment_info[0]['imaging_frame_rate']
        frames_per_folder = self._get_frames_per_folder()
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
                    point_idx = stim_idx % len(acq_info['point_order'])
                    if point_idx == 0:
                        point_idx = len(acq_info['point_order'])
                    
                    # Record stimulation
                    abs_frame = frame_idx + current_start_frame
                    stim_id[abs_frame] = acq_info['point_order'][point_idx]
                
                current_start_frame = block_edges[acq_num]

        # Create trial windows (removed hardcoded values)
        trial_frames, full_trials = self._create_trial_windows(
            stim_id, points, pre_trial_window, post_trial_window, block_edges
        )

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
        """Save analysis results"""
        save_path = os.path.join(self.save_folder, 'trial_and_target_info.npz')
        np.savez(save_path, **kwargs)

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
            dist = dist * micronsperpix
            
            # Find cells within 20 microns
            min_ind = np.where(dist < 20)[0]
            target_id.append(cell_ind[min_ind])
            dist_id.append(dist[min_ind])
        
        # Create figure showing ROIs
        img = (ops['max_proj'] - ops['max_proj'].min()) / (ops['max_proj'].max() - ops['max_proj'].min())
        
        for i in range(len(points)):
            plt.clf()
            plt.imshow(img)
            plt.scatter(points[i, 0], points[i, 1], s=100, c='b', marker='o')
            
            # Plot ROI boundaries for each target
            for n in range(len(target_id[i])):
                try:
                    self._plot_mask_boundaries(target_id[i][n], stat, color='r')
                except:
                    continue
                    
            plt.savefig(os.path.join(self.save_folder, f'{i+1} roi check.png'))
            plt.close()
        
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
        """Parse XML metadata file"""
        """
        Args:
            xml_file (str): Path to XML metadata file

        Returns:
            dict: Parsed metadata
        """
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_file)
        root = tree.getroot()

        return root
    
    def _load_sync_data(self, sync_file):
        """Load sync data from h5 file"""
        with h5py.File(sync_file, 'r') as f:
            sync_data = f['data'][:]
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
    
    def _smooth(self, input_data, window, kernel_type='boxcar'):
        """
        Smoothing function with gaussian or boxcar kernel.
        
        Args:
            input_data (np.ndarray): Input signal (must be one-dimensional)
            window (int): Total kernel width
            kernel_type (str): 'gauss' for gaussian kernel or 'boxcar' for simple moving window average
            
        Returns:
            np.ndarray: Smoothed signal
            
        Raises:
            ValueError: If input dimensions or window size are invalid
        """
        # Check input dimensions
        input_dims = input_data.ndim
        input_size = input_data.shape
        if input_dims > 2 or min(input_size) > 1:
            raise ValueError('Input array is too large.')

        # Validate window size
        if window < 1 or window != round(window):
            raise ValueError('Invalid smooth window argument')

        # Early return for window size 1
        if window == 1:
            return input_data

        # Handle binary kernel type (not implemented, would need bin_data function)
        if kernel_type.startswith('bin'):
            raise NotImplementedError("Binary kernel type not implemented")

        # Transpose if needed
        if len(input_size) > 1 and input_size[1] > input_size[0]:
            input_data = input_data.T
            toggle_dims = True
        else:
            toggle_dims = False

        # Ensure window is even
        if window/2 != round(window/2):
            window = window + 1
        halfwin = window//2

        input_length = len(input_data)

        # Create kernel
        if kernel_type.startswith('gauss'):
            x = np.arange(-halfwin, halfwin+1)
            kernel = np.exp(-x**2/(window/2)**2)
        else:
            kernel = np.ones(window)
        kernel = kernel/np.sum(kernel)

        # Create padded array with mean values at edges
        mn1 = np.mean(input_data[:halfwin])
        mn2 = np.mean(input_data[-halfwin:])
        
        padded = np.zeros(input_length + 2*halfwin)
        padded[halfwin:input_length+halfwin] = input_data
        padded[:halfwin] = mn1
        padded[input_length+halfwin:] = mn2

        # Convolve and trim
        output = np.convolve(padded, kernel, mode='full')
        output = output[window:input_length+window]

        # Restore original dimensions if needed
        if toggle_dims:
            output = output.T

        return output