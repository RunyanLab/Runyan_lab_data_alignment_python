
import numpy as np
import pickle
import os
from scipy.io import savemat

class CalculateInfluence:
    def __init__(self):
        # Initialize paths
        self.raw_data_folder = None
        self.suite2p_folder = None
        self.sync_base_path = None
        self.save_folder = None
        
    def inf_calculation(self, trial_frames_final, stim_id, activity_struct, pre, post, full_trials):

        structs = {(i,j): None for i in range(len(activity_struct)) for j in range(len(trial_frames_final))}
        # Main loop
        for ind in range(len(activity_struct)): # Loop through activity struct
            for target in range(len(trial_frames_final)):  # Loop through targets

                # +1 to differentiate first target from zeros in array
                stim_locs = np.where(stim_id == target + 1)[0]  
                structs[ind, target] = np.zeros([activity_struct[ind].shape[0] ,trial_frames_final[target].shape[0] ,len(trial_frames_final[target][0, :])])

                for trial in range(trial_frames_final[target].shape[0]):  # Loop through trials
                    # Check if trial has full frames for influence calculation and to ensure matching dimensions 
                    if not full_trials[target][:][trial]:
                        continue

                    # Pull out trial frames
                    trial_frames = np.asarray(trial_frames_final[target][trial, :],dtype=int)

                    # Double check if trial frames goes beyond max imaging frames 
                    if np.max(trial_frames) < activity_struct[ind].shape[1]:
                        # Store activity traces 
                        structs[ind, target][:, trial, :] = activity_struct[ind][:, trial_frames]

        influence = {(i) : None for i in range(len(activity_struct))} 
        for ind in range(len(activity_struct)):
            
            num_targets = len(trial_frames_final)
            num_neurons = structs[ind,0].shape[0]
            influence[ind] =np.zeros([num_targets, num_neurons], dtype=float)
            influence[ind] =np.zeros([num_targets, num_neurons], dtype=float)

            for target in range(len(trial_frames_final)):
                #     current = structs[ind, target]
                #     num_neurons, num_trials, num_frames = current.shape

                current = structs[ind,target]
                num_trials = current.shape[1]
                
                delta = np.zeros([num_neurons,num_trials], dtype=float)
                
                for neuron in range(num_neurons):
                    pre_act = current[neuron, :, pre-31:pre]
                    post_act = current[neuron, :, post:31+post]

                    # Calculate delta for this neuron
                    delta[neuron, :] = np.mean(post_act, axis=1) - np.mean(pre_act, axis=1) 
                    
                # Calculate influence
                mean_delta = np.mean(delta[:,:], axis=1)
                std_delta = np.std(delta[:,:], axis=1, ddof=0)  # ddof=0 for population std
                influence[ind][target,:] = mean_delta / std_delta

        return influence, structs

    def calculate_max_motion(self, dff, trial_frames, distoff_um):  
        xyMotionLimit = 5  # x/y motion limit in microns
        the_nans = np.where(np.isnan(dff[0, :]))[0]  # Find indices of NaNs in the first row of dffnon
        trial_frames_final = [tf.copy() for tf in trial_frames]  # Make a copy of trial_frames

        # Motion correction
        for n in range(len(trial_frames)):
            tmp = []  # List to store trials to be removed
            for tr in range(trial_frames[n].shape[0] - 1):
                # Check if the maximum motion in the trial exceeds the limit
                inds = trial_frames_final[n][tr, :]
                inds = np.asarray(inds,dtype=int)
                if np.max(distoff_um[inds]) >= xyMotionLimit:
                    tmp.append(tr)
        
            trial_frames_final[n] = np.delete(trial_frames_final[n], tmp, axis=0)

        return trial_frames_final
        
    def get_active_inds(self, s2p_active_inds, red_cell_index, non_cell_locs, points):
        """
        Convert MATLAB code to find active indices in Python.

        Parameters:
        s2p_active_inds (list or np.ndarray): Input indices, possibly containing NaNs.
        red_cell_index (list or np.ndarray): Array to search for matches.

        Returns:
        np.ndarray: Active indices array with matches or NaNs.
        """
        # Initialize active_inds with NaNs
        active_inds = np.full(len(s2p_active_inds), np.nan)

        for i, val in enumerate(s2p_active_inds):
            if not np.isnan(val):  # Check if the value is not NaN
                try:
                    # Find the index of the value in red_cell_index
                    active_inds[i] = np.where(red_cell_index == val)[0][0]
                except IndexError:
                    # If no match is found, leave it as NaN
                    active_inds[i] = np.nan
                # Loop over unique active indices
        for i in np.unique(active_inds):
            if not np.isnan(i):  # Skip NaN values
                inds = np.where(active_inds == i)[0]  # Find occurrences of the current index
                if len(inds) > 1:  # Check if there are duplicates
                    tmp = np.flip(non_cell_locs[int(active_inds[inds[0]]), :].astype(float))  # Reverse and convert to float
                    d = []
                    for n in inds:
                        distance = np.sqrt(np.sum((tmp - points[n, :]) ** 2, axis=0))
                        d.append(distance)
                    
                    # Sort distances and retain only the closest
                    sorted_indices = np.argsort(d)
                    active_inds[inds[sorted_indices[1:]]] = np.nan  # Mark all but the closest as NaN

        # Create idx with non-NaN active indices
        idx = np.arange(points.shape[0])  # Start with all indices
        idx = idx[~np.isnan(active_inds)]  # Exclude indices where active_inds is NaN
        return active_inds, idx 

    def calculate_distances(self, active_inds, cell_locs, cell_type, points, microns_per_pix, save_folder):
        cell_locs = np.array(cell_locs, dtype=float)
        points = np.array(points, dtype=float)
        active_inds = np.array(active_inds, dtype=float)

        num_targets = len(active_inds)
        cell_type = np.array(cell_type, dtype=float)
        num_non_cells = np.sum(cell_type == 0)
        non_inds = cell_type==0 
        red_inds = cell_type==1
        num_red_cells = np.sum(cell_type == 1)

        # Initialize distance arrays
        dist_from_target = np.zeros((num_targets, num_non_cells))
        dist_from_target_red = np.zeros((num_targets, num_red_cells))

        # Compute distances for targets
        for i in range(num_targets):
            if not np.isnan(active_inds[i]):  # If the target has an ROI
                cel1 = int(active_inds[i])
                dist_from_target[i, :] = np.sqrt(
                    np.sum((cell_locs[cel1, :] - cell_locs[non_inds,:])**2, axis=1)
                )
                dist_from_target_red[i, :] = np.sqrt(
                    np.sum((cell_locs[cel1, :] - cell_locs[red_inds,:])**2, axis=1)
                )
            else:  # No ROI associated
                cel1 = i
                dist_from_target[i, :] = np.sqrt(
                    np.sum((points[cel1, :] - cell_locs[non_inds,:])**2, axis=1)
                )
                dist_from_target_red[i, :] = np.sqrt(
                    np.sum((points[cel1, :] -  cell_locs[red_inds,:])**2, axis=1)
                )
        # Convert distances from pixels to microns
        dist_from_target *= microns_per_pix
        dist_from_target_red *= microns_per_pix

        dist_redred = np.sqrt(
            np.sum((cell_locs[red_inds, None, :] - cell_locs[None, red_inds, :])**2, axis=2)
        ) * microns_per_pix

        dist_nonnon = np.sqrt(
            np.sum((cell_locs[non_inds, None, :] - cell_locs[None, non_inds, :])**2, axis=2)
        ) * microns_per_pix

        dist_nonred = np.sqrt(
            np.sum((cell_locs[red_inds, None, :] - cell_locs[None, non_inds, :])**2, axis=2)
        ) * microns_per_pix

        # Combine all distances into one array
        dist_all = np.block([
            [dist_redred, dist_nonred],
            [dist_nonred.T, dist_nonnon]
        ])

        # Package distances into a dictionary
        dist = {
            'redred': dist_redred,
            'nonnon': dist_nonnon,
            'nonred': dist_nonred,
            'all': dist_all
        }
        np.savez_compressed(
                    f"{save_folder}/dist",
                    dist=dist,
                    dist_from_target=dist_from_target,
                    dist_from_target_red=dist_from_target_red
                )

        return dist, dist_from_target, dist_from_target_red
            
    def create_cell_locs_struct(self, save_folder, stat,iscell):
        with open(f'{save_folder}/indices.pkl', 'rb') as file:
            indices = pickle.load(file)

        red_inds = indices['s2p_indices']['red_cell_inds']
        non_inds = indices['s2p_indices']['non_cell_inds']

        cell_locs = [] 
        cell_type = []

        for cell in range(len(stat)):
            if iscell[cell][0] == 1:
                cell_locs.append(stat[cell]['med'])
                if cell in red_inds:
                    cell_type.append(1)
                elif cell in non_inds:
                    cell_type.append(0)
        return cell_locs, cell_type

    def save_structures(self,base, distances,inf_data,activity):
        import hdf5storage
        name = os.path.join(base,'distances.pkl')
        with open(name,"wb") as file:
            pickle.dump(distances,file,)
        
        name = os.path.join(base,'distances.mat')
        hdf5storage.savemat(name,distances, format='7.3', oned_as='column',truncate_existing=True)

        name = os.path.join(base,'inf_data.pkl')
        with open(name,"wb") as file:
            pickle.dump(inf_data,file)

        name = os.path.join(base,'inf_data.mat')
        hdf5storage.savemat(name,inf_data, format='7.3', oned_as='column',truncate_existing=True)

        name = os.path.join(base,'activity.pkl')
        with open(name,"wb") as file:
            pickle.dump(activity,file)
        
        name = os.path.join(base,'activity.mat')
        hdf5storage.savemat(name,activity, format='7.3', oned_as='column',truncate_existing=True)

    def inf_calculation_control_stims(self, trial_frames_final, stim_id, activity_struct, pre, post, full_trials,target_cells):
        structs = {(i,j): None for i in range(len(activity_struct)) for j in range(len(trial_frames_final))}
        # Main loop
        for ind in range(len(activity_struct)): # Loop through activity struct
            for target in range(len(trial_frames_final)):  # Loop through targets

                # +1 to differentiate first target from zeros in array
                stim_locs = np.where(stim_id == target + 1)[0]  
                structs[ind, target] = np.zeros([activity_struct[ind].shape[0] ,trial_frames_final[target].shape[0] ,len(trial_frames_final[target][0, :])])

                for trial in range(trial_frames_final[target].shape[0]):  # Loop through trials
                    # Check if trial has full frames for influence calculation and to ensure matching dimensions 
                    if not full_trials[target][:][trial]:
                        continue

                    # Pull out trial frames
                    trial_frames = np.asarray(trial_frames_final[target][trial, :],dtype=int)

                    # Double check if trial frames goes beyond max imaging frames 
                    if np.max(trial_frames) < activity_struct[ind].shape[1]:
                        # Store activity traces 
                        structs[ind, target][:, trial, :] = activity_struct[ind][:, trial_frames]

            influence = {(i) : None for i in range(len(activity_struct))} 

        for ind in range(len(activity_struct)):
            
            num_targets = len(trial_frames_final)
            num_neurons = structs[ind,0].shape[0]
            influence[ind] =np.zeros([num_targets, num_neurons], dtype=float)
            influence[ind] =np.zeros([num_targets, num_neurons], dtype=float)

            for target in range(len(trial_frames_final)):
                #     current = structs[ind, target]
                #     num_neurons, num_trials, num_frames = current.shape

                current = structs[ind,target]
                num_trials = current.shape[1]
                
                after = np.zeros([num_neurons,num_trials], dtype=float)
                
                for neuron in range(num_neurons):
                    #pre_act = current[neuron, :, pre-31:pre]
                    post_act = current[neuron, :, post:31+post]

                    # Calculate delta for this neuron
                    #delta[neuron, :] = np.mean(post_act, axis=1) - np.mean(pre_act, axis=1) 
                    after[neuron,:] = np.mean(post_act, axis=1)

                # Calculate influence
                # mean_delta = np.mean(delta[:,:], axis=1)
                # std_delta = np.std(delta[:,:], axis=1, ddof=0)  # ddof=0 for population std
                # influence[ind][target,:] = mean_delta / std_delta
            

        return influence, structs

    #def create_trial_frames_control_stims(imaging):

