%% load the data

%CAR changed this so that the code isn't constantly getting changed
rawDataFolder = uigetdir('This PC','Pick Directory Containing 2P Acquisitions');
suite2pFolder = uigetdir('This PC','Pick Directory Containing Suite2P Files');
sync_base_path = uigetdir('This PC', 'Pick Directory Containing Sync Files');
saveFolder = uigetdir('This PC','Pick Directory to Save These Analysis Files');


load([suite2pFolder '\Fall_red.mat'],'stat','ops','F','spks','Fneu');
% mkdir(saveFolder);
cd(saveFolder)
s2p_badframes=find(ops.badframes>0);

%% loading meta data 
searchEnv = dir([rawDataFolder,filesep,'**',filesep,'*.xml']);
frames_per_folder=getFramesPerFolder(rawDataFolder,2);
block_edges=cumsum(frames_per_folder);
block_edges(block_edges==0)=[];

%% finding stimulation times and locations

channel_number = 5; %the channel number of the imaging y galvo 
plot_on = 1; %set to 0 if you want to supress the plots created 
%%to do: why are we using a different get_frame_times function now?
[alignment_info] = get_frame_times_no_LED(rawDataFolder, sync_base_path, channel_number,plot_on, frames_per_folder);%%%why do we have a "no_LED" version of this?

[spiral_alignment_info, metaData, markPoints] = get_spiral_locations_and_times_car(alignment_info,rawDataFolder, sync_base_path, ops);
micronsperpix = spiral_alignment_info(end).microns_per_pixel; 
xoff=ops.xoff;
yoff=ops.yoff;
%% Calculating the point order and saving the ROIs near each target
%
[xoff_um,yoff_um,distoff_um]=getOffsetInfo(xoff,yoff,micronsperpix);
save([saveFolder '\offsetInfo'],'xoff','yoff','xoff_um','yoff_um','distoff_um')

%%to do: pointOrder and points needs to be structures with a point order/points for each
%%acquisition. Also, why do we need both 'points' and 'pointOrder'? how do
%%they differ?

%%%what we want to do now instead of just picking the points from the first
%%%acquisition, is get a full list of all points
temp_points = [];
temp_point_order = [];
for acq = 1:length(spiral_alignment_info)
    temp_points = cat(1,temp_points,spiral_alignment_info(acq).points); 
    temp_point_order = cat(1,temp_point_order,unique(spiral_alignment_info(acq).point_order));  %%be sure this works as it should
end

pointOrder = unique(temp_point_order);
for p = 1:length(pointOrder)
    ind = find(temp_point_order==pointOrder(p),1,'last');
    points(p,:) = temp_points(ind,:);
end


mkdir([saveFolder '\roi plots\']);
[targetID,distID] = targetFind(points, stat ,micronsperpix,ops,[saveFolder '\roi plots\']);
try
    fneu_chan2=readNPY([suite2pFolder '\Fneu_chan2.npy']);
catch
    fneu_chan2=readNPY([suite2pFolder '\F_chan2.npy']);
end

%% creating the trial frames 
framerate = alignment_info(1).imaging_frame_rate; 
stim_id=zeros(1,sum(frames_per_folder));
current_start_frame = 0; 
for acq = 1 : length(alignment_info)
    if isempty(spiral_alignment_info(acq).points)==0
    block_frames{acq}=(block_edges(acq)-frames_per_folder(acq)+1):block_edges(acq);
    for stim = 1 : length(spiral_alignment_info(acq).spiral_start_times)
        % because the stim time isn't aligned perfectly to the frame times,
        % must find the frame closest to the wavesurfer unit of the stim
        % time and assign that as the stim frame 
        [val,ind]=min(abs(alignment_info(acq).frame_times-spiral_alignment_info(acq).spiral_start_times(stim)));
        stim_temp(1,stim) = ind; %%%time of the stimulation

        % this method works if each acq has the same point order 
        % keeps track of a counter and mods it by length of point order to
        % find current point 
        %%%%to do: update with the point order & point identities of this specific acquisition
        indexx=mod(stim,length(spiral_alignment_info(acq).point_order));
        if indexx == 0
            indexx = length(spiral_alignment_info(acq).point_order); 
        end 
        stim_temp(2,stim) = spiral_alignment_info(acq).point_order(indexx);
    end
    stim_temp(1,:) = stim_temp(1,:) + current_start_frame; 
    stim_id(stim_temp(1,:)) = stim_temp(2,:); 
    current_start_frame = block_edges(acq); 
    clear('stim_temp') 
    end
end 

% code doesn't use this now, but could be helpful to assign post trial
% window 
time_between_stim = find(stim_id,2);
time_between_stim = time_between_stim(1,2) - time_between_stim(1,1);

stim_on = zeros(size(stim_id));
stim_on(stim_id~=0) = 1; 

[~,locs] = findpeaks(stim_on); 
for ii = 1 : length(locs)-1
    time_between(ii) = locs(ii+1) - locs(ii);
end 

%65 frames post stim & 65 frames pre stim for trial_frames, determined
%manually, make sure this works for the time between stimulations 
% pre_trial_window = round(time_between_stim/2); this method worked for
% regular stimulations. DOES NOT work for triggered ones.
% post_trial_window = time_between_stim; 
pre_trial_window = 30;
post_trial_window = 30;
trial_frames = cell(1,length(points));
full_trials = cell(1,length(points));

for i = 1 : length(trial_frames)
    index = find(stim_id==i);
    trial_frames{1,i} = zeros(length(index),post_trial_window + pre_trial_window+1);
    full_trials{1,i} = zeros(length(index),1); 

    for ii = 1 : length(index)-1
        time_between(ii) = locs(ii+1) - locs(ii);
    end

    for trial = 1 : length(index)
        if ~(index(trial)>pre_trial_window) || time_between(trial) < post_trial_window
            trial_frames{1,i}(trial,:) = (index(trial):index(trial)+post_trial_window+pre_trial_window); %%this is a placeholder for now, due to the issue of the bad trigger immediately after starting a tseries.
            full_trials{1,i}(trial) = 0;
        else
            trial_frames{1,i}(trial,:) = (index(trial)-pre_trial_window : index(trial)+post_trial_window);
            full_trials{1,i}(trial) = 1;
        end

        block = find(index(trial) < block_edges,1);
        if sum(ismember(trial_frames{1,i}(trial,:),block_frames{block})) == post_trial_window+pre_trial_window+1
            trial_block_id{i}(trial)=block;
        else
            full_trials{1,i}(trial) = 0;
        end


        % for b=1:length(block_edges)
        % 
        %     if length(find(ismember(trial_frames{1,i}(trial,:),block_frames{b})))==post_trial_window + pre_trial_window+1
        %         trial_block_id{i}(trial)=b;     
        %     end
        % end


    end
end


for i = 1 : length(block_edges)
    trial_num(i) = length(spiral_alignment_info(i).spiral_start_times);
end 

%% Saving candidate ROI plots of activity
for roi=1:size(F,1)
    for target_id=1:size(points,1)
        for tr=1:size(trial_frames{target_id},1)
            if max(trial_frames{target_id}(tr,:))<size(F,2)
            F_struct{target_id}(tr,:,roi)=F(roi,trial_frames{target_id}(tr,:));
            F_struct2{target_id}(tr,:,roi)=F(roi,trial_frames{target_id}(tr,:))-.7*Fneu(roi,trial_frames{target_id}(tr,:));
            Fneu_struct{target_id}(tr,:,roi)=Fneu(roi,trial_frames{target_id}(tr,:));
            end
        end
    end
end

figure
for i=1:size(points,1)
    clf
    [~,order]=sortrows(distID{i});
    for n=1:size(order,1)
        subplot(1,length(targetID{i}),n)
        hold on
        tmp=[];
        for tr=1:size(F_struct2{i},1)
            tmp=cat(1,tmp,F_struct2{i}(tr,:,targetID{i}(order(n))));
%           plot(F_struct2{i}(tr,:,targetID{i}(n)))
        end
        plot(mean(tmp),'k','linewidth',2)
        title(num2str(targetID{i}(order(n))))

    end
    saveas(gcf,[saveFolder '\roi plots\' num2str(i) ' candidate rois neu correct'])
%   pause
end


%% Suite2p target cell selection
% open up suite2p and select the target ROIs based on the candidate plots,
% check the readme for more detailed instructions 
for i=1:size(points,1)
    close all
    openfig([saveFolder '\roi plots\' num2str(i) ' candidate rois neu correct.fig'])
    openfig([saveFolder '\roi plots\' num2str(i) ' roi check.fig'])
    caxis([0 .25])
    pause
end

%% Section 9 
t=load([suite2pFolder '\Fall_target.mat'],'iscell');
potential_targets=find(t.iscell(:,1)==1);
s2p_active_inds=[];
for i=1:size(points,1)
    
    if ~isempty(find(ismember(targetID{i},potential_targets),1))
        
        tmp_idx=find(ismember(targetID{i},potential_targets));
        if length(tmp_idx>1)
            tmp_idx=tmp_idx(1);
        end

        s2p_active_inds(i)=targetID{i}(tmp_idx);
    else
        s2p_active_inds(i)=nan;
    end
end

for i=1:length(s2p_active_inds)
    if ~isnan(s2p_active_inds(i))
        clf
%         figure
        hold on
        for b=1:length(block_edges)
            tr=find(trial_block_id{i}==b);
            tmp=[];
            for n=tr
                tmp=[tmp; F_struct2{i}(n,:,s2p_active_inds(i))];
            end
            plot(mean(tmp));
            legend
            title(num2str(i));
            
        end
    pause
    end
end
close all
%  %eg=32
%  eg=30;
%  for b=1:length(block_edges)
%      tr=find(trial_block_id{eg}==b);
%      figure
%      hold on
%      for n=tr
%          plot(F_struct2{eg}(n,:,s2p_active_inds(eg)));
%      end
%      title(num2str(b))
%      pause
%  end
%  close all
%% Section 10 (section # corresponding to christine's version) 
% saving important variables, can add additional variables created in the
% script to look at later
stim_on = double(stim_id(1,:)>0);
save([saveFolder '\trial_and_target_info'],'trial_frames','block_edges','block_frames','framerate','frames_per_folder','micronsperpix',...
    'markPoints','metaData','s2p_active_inds','stim_id','stim_on','trial_num','points','spiral_alignment_info',...
    'rawDataFolder','saveFolder', 'full_trials')

%% Section 11 
%     %%identify bad frames by finding points in raw fluorescence where it
% % %%appears meniscus may have dropped off or there is excessive z-motion.
% % % %%Confirm by looking at tif or .bin image at that timepoint
% % badframes=[27275:36550 120025:129300];
% badframes=[30985:52693];
badframes = [];
badframes=[badframes s2p_badframes];
[dffred, red_cell_index ,red_cell_locs,dffnon, non_cell_index,non_cell_locs,single_inds] = ckDff_python(badframes,suite2pFolder);

save([saveFolder '\dff_raw'],'dffred','red_cell_index','red_cell_locs','dffnon','non_cell_index','non_cell_locs','single_inds');

eliminate_cells_raw_F_single_cell(s2p_active_inds, suite2pFolder, saveFolder); 

