%% get influence 
function get_influence_JM(suite2pFolder,saveFolder)

neuropilPercent = .7; % portion of neuropil which is subtracted from F_full, .7 is the suite2p convention 

% suite2pFolder = uigetdir('This PC','Pick Directory Containing Suite2P Files');
% saveFolder = uigetdir('This PC','Pick Directory to Save These Analysis Files');
saveFolder = [saveFolder '/'];

dataFolder= saveFolder;
load([saveFolder '/trial_and_target_info.mat'])
load([suite2pFolder '/Fall.mat'],'F','Fneu','ops');
load([dataFolder '/trial_and_target_info.mat'],'block_edges','micronsperpix',...
    'points','s2p_active_inds','trial_frames','stim_id','full_trials');
load([dataFolder '/offsetInfo']);
load([dataFolder '/dff_elim']);
load([dataFolder '/deconv.mat']);

F_full=F-neuropilPercent*Fneu;          %%%why is this done here, and not before deconvolution?    
F_non=F_full(non_cell_index,:);
F_red=F_full(red_cell_index,:);
clear F_full F Fneu


%% Motion Correction
% finding frames where x,y motion is greater than 5 microns and setting
% those frames to nans, large x,y motions leds to uncertainty in stim
% targeting
xyMotionLimit = 5; %x/y motion limit in microns
the_nans=find(isnan(dffnon(1,:)));
trial_frames_final=trial_frames;
for n=1:size(trial_frames,2)
    tmp=[];
    for tr=1:size(trial_frames{n},1)-1
        if max(distoff_um(trial_frames_final{n}(tr,:)))>=xyMotionLimit  
            tmp=[tmp tr];
%         elseif length(find(ismember(trial_frames{n}(tr,:),block_edges)))>0 || length(find(ismember(trial_frames{n}(tr,:),the_nans)))>0
%             tmp=[tmp tr];  %%taking this out temporarily
        end
        
    end
    trial_frames_final{n}(tmp,:)=[];
end
%% 
% s2p_active_inds is the indexs of the targets in relation to their position
% in the suite2p index (index in total set of ROIs). active_inds is the indexs of the targets in the
% non_cell_index and locs arrays (their index in the set of ROIs that are
% cells) 
% This section creates the active_inds variable for indexing in the set of
% ROI's which are cells
active_inds = zeros(1,length(s2p_active_inds));
for i=1:length(s2p_active_inds)
    if ~isnan(s2p_active_inds(i))
        try
            active_inds(i)=find(red_cell_index==s2p_active_inds(i));
        catch
            active_inds(i)=nan;
        end
    else
        active_inds(i)=nan;
    end
end

% checking for double assignment two targets to being the same ROI. Assigns
% ROI to closest target
for i=unique(active_inds)
    try
        if length(find(active_inds==i))>1
            inds=find(active_inds==i);
            d=[];
            tmp=double(fliplr(non_cell_locs(active_inds(inds(1)),:)));
            for n=inds
                d=[d sqrt(sum((tmp-points(n,:)).^2,2))];
            end
            [~,sorted]=sort(d);
            active_inds(inds(sorted(2:end)))=nan;
        end
    end
end
idx=1:size(points,1);
idx(find(isnan(active_inds)))=[]; %idx is the index's of targeted cells in the set of ROI's that are cells. Not including NaNs of any kind. 

%% calculate influence

post = input('Number of frames post-stim to use in influence calculation?')
endOfBlockFrame = stimFrame+post; %last frame consider in the post stim block
pre = input('Number of frames pre-stim to use in influence calculation?')

activity_struct = {deconvnon, deconvred}; 
pre = 15; post = 15; 
[influence_deconv] = inf_calculation_JM(trial_frames_final, stim_id, activity_struct, pre, post,full_trials);

activity_struct = {dffnon, dffred}; 
pre = 15; post = 15; 
[influence_dff] = inf_calculation_JM(trial_frames_final, stim_id, activity_struct, pre, post,full_trials); 


%% find control indexes 

% finding control indexes defined as having low average influence, .75
% arbitary value could be changed based on preference 
% UNCOMMENT FOR EXCITATORY STIM PROC - uncomment the lines in the crtl
% calculations
% minMaxInfluence = .75; 
% 
% ctrl_idx=find(max([influence_red ; influence_non])<minMaxInfluence); 
% % red_ctrl_idx=ctrl_idx(find(ismember(ctrl_idx,red_points)));  %%commented
% % out CAR 08/09/2023
% red_ctrl_idx = [];  %%added CAR 08/09/2023
% bad_target=find(ismember(idx,ctrl_idx));
% for n=1:size(influence_non,2) 
%     try
%         if influence_non(active_inds(n),n)<.5
%             bad_target=[bad_target n];
%         end
%     end
% end
% active_inds_deconv=active_inds;
% %active_inds_deconv(bad_target)=nan;
% idx=1:size(points,1); %reusing variable? 
% idx(find(isnan(active_inds_deconv)))=[];
% 
% 
% 
% ctrl_idx_dff1=find(max([influence_red_dff ; influence_non_dff])<minMaxInfluence);
% % red_ctrl_idx=ctrl_idx(find(ismember(ctrl_idx,red_points)));
% red_ctrl_idx = [];
% bad_target=find(ismember(idx,ctrl_idx_dff1));
% for n=1:size(influence_non_dff,2)
%     try
%         if influence_non_dff(active_inds(n),n)<.5
%             bad_target=[bad_target n];
%         end
%     end
% end
% active_inds_dff1=active_inds;
% %active_inds_dff1(bad_target)=nan;
% idx_dff1=1:size(points,1);
% idx_dff1(find(isnan(active_inds_dff1)))=[];
% 
% ctrl_idx_dff2=find(max([influence_red_dff_new_b ; influence_non_dff_new_b])<minMaxInfluence);
% % red_ctrl_idx=ctrl_idx(find(ismember(ctrl_idx,red_points)));
% red_ctrl_idx = [];
% bad_target=find(ismember(idx,ctrl_idx_dff2));
% for n=1:size(influence_non_dff_new_b,2)
%     try
%         if influence_non_dff_new_b(active_inds(n),n)<.5
%             bad_target=[bad_target n];
%         end
%     end
% end
% active_inds_dff2=active_inds;
% %active_inds_dff2(bad_target)=nan;
% idx_dff2=1:size(points,1);
% idx_dff2(find(isnan(active_inds_dff2)))=[];

%% 


save([saveFolder '/trial_frames_final'],'trial_frames_final');
save([saveFolder '/influence_info_deconv'],'active_inds','active_inds_deconv', 'ctrl_idx','deconvnon_trial_struct','deconvred_trial_struct','idx',...
    'influence_non','influence_red')

save([saveFolder '/influence_info_dff1'],'active_inds','active_inds_dff1', 'ctrl_idx_dff1','dffnon_trial_struct','dffred_trial_struct','idx_dff1',...
    'influence_non_dff','influence_red_dff')

save([saveFolder '/influence_info_dff2'],'active_inds','active_inds_dff2', 'ctrl_idx_dff2','dffnon_new_b_trial_struct','dffred_new_b_trial_struct','idx_dff2',...
    'influence_non_dff_new_b','influence_red_dff_new_b')

%% Distance calculations 
non_cell_locs=double(non_cell_locs);
red_cell_locs=double(red_cell_locs);
for i=1:size(influence_non,2) % loop targets 
    if ~isnan(active_inds(i)) %if the target has an ROI associated with it, then the distance will be centered on the center of the ROI 
        cel1=active_inds(i);
        for cel2=1:size(deconvnon,1) % loop non red cells 
            dist_from_target(i,cel2)=sqrt((non_cell_locs(cel1,1)-non_cell_locs(cel2,1))^2+(non_cell_locs(cel1,2)-non_cell_locs(cel2,2))^2);
        end
        for cel2=1:size(dffred,1) % loop red cells 
            dist_from_target_red(i,cel2)=sqrt((non_cell_locs(cel1,1)-red_cell_locs(cel2,1))^2+(non_cell_locs(cel1,2)-red_cell_locs(cel2,2))^2);
        end
    else
        cel1=i;
        for cel2=1:size(deconvnon,1)
            dist_from_target(cel1,cel2)=sqrt((points(cel1,1)-non_cell_locs(cel2,1))^2+(points(cel1,2)-non_cell_locs(cel2,2))^2);
        end
        for cel2=1:size(dffred,1)
            dist_from_target_red(cel1,cel2)=sqrt((points(cel1,1)-red_cell_locs(cel2,1))^2+(points(cel1,2)-red_cell_locs(cel2,2))^2);
        end
    end
end
dist_from_target=dist_from_target*micronsperpix; %convert distance from pixels to microns 
dist_from_target_red=dist_from_target_red*micronsperpix;

cels1=size(red_cell_locs,1);
cels2=size(non_cell_locs,1);

for cel1=1:cels1 % loop red cells 
    for cel2=1:cels1 % loop red cells 
        dist.subsub(cel1,cel2)=sqrt((red_cell_locs(cel1,1)-red_cell_locs(cel2,1))^2+(red_cell_locs(cel1,2)-red_cell_locs(cel2,2))^2);
    end
end
dist.subsub=dist.subsub*micronsperpix;
for cel1=1:cels2 % loop non red cells 
    for cel2=1:cels2 % loop non red cells 
        dist.nonnon(cel1,cel2)=sqrt((non_cell_locs(cel1,1)-non_cell_locs(cel2,1))^2+(non_cell_locs(cel1,2)-non_cell_locs(cel2,2))^2);
    end
end
dist.nonnon=dist.nonnon*micronsperpix;

for cel1=1:cels1 % loop red cells 
    for cel2=1:cels2 % loop non red cells 
        dist.nonsub(cel1,cel2)=sqrt((red_cell_locs(cel1,1)-non_cell_locs(cel2,1))^2+(red_cell_locs(cel1,2)-non_cell_locs(cel2,2))^2);
    end
end
dist.nonsub=dist.nonsub*micronsperpix;

dist.all=[dist.subsub dist.nonsub; dist.nonsub' dist.nonnon];        
        
        
save([saveFolder '/dist'],'dist','dist_from_target','dist_from_target_red')        
        
        
        
        
        
        
