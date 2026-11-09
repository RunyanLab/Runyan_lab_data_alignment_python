# **Runyan_lab_data_alignment_python**
A python based version of the neural and behavioral data alignment process for the Runyan lab

## **Purpose of the Processing Pipeline**
This pipeline processes 2-photon calcium imaging data and combines the imaging data with VIRMEN behavioral information into an aligned and trialized datastructure. 
Delta F/F is calculated on rawF traces from Suite2p, and the Delta F/F traces are then deconvolved. 
Alignment of the imaging frames and the VIRMEN trials is achieved through sync information recorded by ClampX. 

## **Installation Instructions**
#### Make sure to clone the MAIN branch in the RunyanLab Organization
##### For Windows
git clone https://github.com/RunyanLab/Runyan_lab_data_alignment_python.git

cd Runyan_lab_data_alignment_python

conda env create -f windows_environment.yml

##### For Mac
git clone https://github.com/RunyanLab/Runyan_lab_data_alignment_python.git

cd Runyan_lab_data_alignment_python

conda env create -f mac_environment.yml

## **Experimental Set up**
#### VIRMEN Behavioral set up 
- This pipeline is built to be used with behavioral data collected with the VIRMEN package in MATLAB. 
- See VIRMEN Github for more information. https://github.com/RunyanLab/VIRMEN_runyanlab 

#### 2-photon imaging
- This pipeline is built to be used with 2photon calcium imaging. It should be generalizable for varying framerates as well as window sizes.
- It has been extensively tested with 512x512 images acquired with a framerate of 30Hz. 

#### Clampx or Wavesurfer
- To align imaging frames with VIRMEN data, the y-galvo of the imaging laser and the VIRMEN iterations outputted by VIRMEN need to be recorded at a common sampling rate. 
- We achieve this by recording both of those data streams in ClampX at 10kHz when imaging at 30Hz. 
- This pipeline should work for both Clampx and wavesurfer files, but it is optmized for clampx files. 


## **Preprocessing Steps**
#### Conversion to TIFs
- Suite2p requires specific input formats to work. Our imaging set up generates raw imagins files which must be first converted to TIFs using Prairieview before running Suite2p


#### Suite2p processing & cell selection 
- This pipeline is written to access the outputs of suite2p. 
- For the exact version of suite2p it has been tested with see https://github.com/RunyanLab/Runyan_lab_suite2p_installation



## **Main processing Script**
#### Dataset specific values
- In this section set the appropriate values for each dataset
- This format works only with specific saving conventions. 
- Processed data should be stored in \\Server\Experimenter_Name\ProcessedData\Mouse\YYYY-MM-DD
- Sync data should be stored in \\Server\Experimenter_Name\ClampX\Mouse\YYYY-MM-DD or \\Server\Experimenter Name\Mouse\ClampX
- VIRMEN data should be stored in \\Server\Experimenter_Name\Mouse\VIRMEN


#### Df/f and Deconvolution
- Df/f is calculated with a rolling window on the rawF traces from suite2p. 
- Deconvolution is computed with the OASIS algorithm.
- The deconvolution class computes dF/F by subtracting and dividing by the 8th percentile baseline from a window of ~30 seconds (the size parameter) around each frame.
- It then deconvolves these dF/F signals using the OASIS toolbox with the AR1 FOOPSI algorithm.
- The code also loads and preprocesses imaging data from suite2p outputs; applies a neuropil correction (subtracting 0.7 * Fneu from F); z-scores the dF/F values
- The class also contains optimized functions using Numba's JIT compilation for parallel processing of percentile calculations, making it computationally efficient for large datasets.

#### Virmen trial alignment
- Alignment of VIRMEN trials with imaging frames is achieved by the VIRMEN iterations recorded in ClampX. 
- A VIRMEN iteration of value -1 is outputted for every iteration of the runtime code, and behavioral data is saved for each loop iteration. 
- Every 10k iterations a positive peak of height equivalent to 10k * pos peak number. 
- The absolute index of each iteration is then calculated by referencing their distance from the two surrounding positive peaks. 

- Imaging frame times come from peaks in the y-galvo signal. Each TSeries' frame count is read from its Bruker `.xml` (one `<Frame>` per frame), so both single-page TIFs (one file per frame) and multi-page OME-TIFs (many frames per file) are handled.
- `align_virmen_data` checks that the total number of frame times matches the number of frames in the suite2p traces and raises an error if they differ (`check_frames=False` skips the check).

#### Troubleshooting: few or no trials aligned
- **Frame count mismatch error / `0 / N trials aligned`**: compare the per-sweep `frames` printed by `get_frame_times_parallel` with the frame counts in each TSeries `.xml` and with suite2p's `ops['nframes']`. These three should agree.
  - Very small frame counts (e.g. 5 per sweep) mean frames were counted from TIF files instead of the XML. Multi-page OME-TIFs hold ~1000 frames per file.
  - A warning that fewer galvo frames were detected than imaging frames usually means the wrong `galvo_channel` or a noisy galvo signal.
- **Blocks of consecutive trials not aligned**: expected when ClampX keeps recording after imaging stops (or VIRMEN starts before the first recording). Those trials have no imaging and get `good_trial = 0`. Compare each ABF's duration with frames / frame rate to confirm.
- **`Read-only file system: '//runyan-fs-01...'` on Mac**: the UNC path only works on Windows. On Mac connect to `smb://runyan-fs-01.bns.pitt.edu/<share>` and use `/Volumes/<share>`.

#### Optional behavior only alignment with clampx
- Option to align VIRMEN trials without imaging frames.
- This can be helpful when troubleshooting alignment issues or to check and see if a dataset will align before image processing has finished. 


## **Outputs and Saved Data**
#### deconv & df/f
- **deconvsig**: deconvolved dff signal. ndArray with shape of [Neurons X Frames]
  
- **dff**: processed F trace. ndArray with shape of [Neurons X Frames]
  
- **z_dff**: Z-scored version of the dff trace. ndArray with shape of [Neurons x Frames]
  
- **celi**: The suite2p indexes relative to their index within the dff array. [1 X number of Cells]  

#### Alignment info
- **Alignment_info**: contains the alignment data such as frame times and sync files used. Optional output to save. Dict of shape [1 x number of acquisitions
  - **keys:**
    -  _imaging_id_ - file name of acquisition
    -  _sync_id_ - file name of sync acquisition
    -  _sync_sampling_rate_ (Hz)
    -  _frame_times_ - time of each frame in pclamp time
    -  _sync_data_ - imaging y-galvo voltage trace  

#### Aligned Trials
- **imaging**: trialized dict containing behavior and imaging data (if it exists) for each VIRMEN trial of the session of shape [1 x number of Trials]
  - **keys for imaging['Trial_n']:**
    -   _start_it_ - starting VIRMEN iteration for trial n
    -   _end_it_ - ending VIRMEN iteration for trial n
    -   _iti_start_it_ - first VIRMEN iteration in the ITI
    -   _iti_end_it_ - last VIRMEN iteration in the ITI
    -   _virmen_trial_info_ - trial conditions (left vs right), choice, and reward
    -   _dff_ - dff matrix for frames of trial n [neurons x frames]
    -   _z_dff_ - z-scored dff matrix for frames of trial n [neurons x frames]
    -   _deconv_ - deconv matrix for frames of trial n [neurons x frames]
    -   _relative_frames_ - frame indexes relative to concatinated matrix of all acquisitions
    -   _file_num_ - acquisition number that contains trial n
    -   _movement_in_virmen_time_ - movement and view angle in VIRMEN time (~60Hz)
    -   _frame_id_ - frame index relative to the frames in the current acquisition. (1-frames in an acquisition)
    -   _movement_in_imaging_time_ - movement and view angle in imaging time (~30Hz)
    -   _good_trial_ - 1 for a trial that has frames for the full length of the trial, 0 if part of the trial was not imaged

- **imaged_trials**: List of trials that have imaging data of shape [1 x number of Trials]

#### Behavior 
- **Behavior**: trialized dict containing behavior information for each VIRMEN trial of shape [1 x number of Trials]
    
     




