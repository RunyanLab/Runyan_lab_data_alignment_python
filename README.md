# **Runyan_lab_data_alignment_python**
A python based version of the neural and behavioral data alignment process for the Runyan lab

## **Purpose of the Processing Pipeline**
This pipeline processes 2-photon calcium imaging data and combines the imaging data with VIRMEN behavioral information into an aligned and trialized datastructure. 
Delta F/F is calculated on rawF traces from Suite2p, and the Delta F/F traces are then deconvolved. 
Alignment of the imaging frames and the VIRMEN trials is achieved through sync information recorded by ClampX. 

## **Installation Instructions**

git clone https://github.com/acbandi213/Runyan_lab_data_alignment_python.git

cd Runyan_lab_data_alignment_python

conda env create -f environment.yml


## **Experimental Set up**
- VIRMEN Behavioral set up
This pipeline is built to be used with behavioral data collected with the VIRMEN package in MATLAB. 
See VIRMEN Github for more information. 

- 2-photon imaging
This pipeline is built to be used with 2photon calcium imaging. It should be generalizable for varying framerates as well as window sizes. 

- Clampx or Wavesurfer
To align imaging frames with VIRMEN data, the y-galvo of the imaging laser and the VIRMEN iterations outputted by VIRMEN need to be recorded at a common sampling rate. 
We achieve this by recording both of those data streams in ClampX at 10kHz when imaging at 30Hz. 
This pipeline should work for both Clampx and wavesurfer files, but it is optmized for clampx files. 


## **Preprocessing Steps**
- conversion to TIFs
Suite2p requires specific input formats to work. Our imaging set up generates raw imagins files which must be first converted to TIFs using Prairieview before running Suite2p


- suite2p processing & cell selection 
This pipeline is written to access the outputs of suite2p. 
For the exact version of suite2p it has been tested with see https://github.com/RunyanLab/Runyan_lab_suite2p_installation



## **Main processing Script**
- Dataset specific values
    In this section set the appropriate values for each dataset
    This format works only with specific saving conventions. 
    Processed data should be stored in \\Server\Experimenter_Name\ProcessedData\Mouse\Date
    Sync data should be stored in \\Server\Experimenter_Name\ClampX\Mouse\Date or \\Server\Experimenter Name\Mouse\ClampX
    VIRMEN data should be stored in \\Server\Experimenter_Name\Mouse\VIRMEN


- Df/f and Deconvolution
    Df/f is calculated with a rolling window on the rawF traces from suite2p. 
    Deconvolution is computed with the OASIS algorithm. 

- Virmen trial alignment
    Alignment of VIRMEN trials with imaging frames is achieved by the VIRMEN iterations recorded in ClampX. 
    A VIRMEN iteration of value -1 is outputted for every iteration of the runtime code, and behavioral data is saved for each loop iteration. 
    Every 10k iterations a positive peak of height equivalent to 10k * pos peak number. 
    The absolute index of each iteration is then calculated by referencing their distance from the two surrounding positive peaks. 

- Optional behavior only alignment with clampx
    Option to align VIRMEN trials without imaging frames.
    This can be helpful when troubleshooting alignment issues or to check and see if a dataset will align before image processing has finished. 


## **Outputs and post processing**
- deconv & df/f 
    deconvsig: deconvolved dff signal. ndArray with shape of [Neurons X Frames]
    dff: processed F trace. ndArray with shape of [Neurons X Frames]
    z_dff: Z-scored version of the dff trace. ndArray with shape of [Neurons x Frames]
    celi: The suite2p indexes relative to their index within the dff array. [1 X number of Cells]  

- Alignment info
    Alignment_info: contains the alignment data such as frame times and sync files used. Optional output to save. Dict of shape [1 x number of acquisitions]

- Aligned Trials
    imaging: trialized dict containing behavior and imaging data (if it exists) for each VIRMEN trial of the session of shape [1 x number of Trials]
    imaged_trials: List of trials that have imaging data of shape [1 x number of Trials]

- Behavior 
    Behavior: trialized dict containing behavior information for each VIRMEN trial of shape [1 x number of Trials]
    
     




