
function [influence] = inf_calculation_JM(trial_frames_final, stim_id, activity_struct, pre, post,full_trials)
%
structs = cell(1,length(activity_struct));

for ind = 1 : length(activity_struct)
    for target=1:size(trial_frames_final,2) %loop targets
        stim_locs = find(stim_id == target);
        for trial=1:size(trial_frames_final{target},1) %loop trials
            if ~full_trials{1,target}(trial,1)
                continue
            end 
            %stim_frame{ind}(target,trial) = find(ismember(trial_frames_final{target}(trial,:), stim_locs));
            structs{ind}{target}(:,:,trial)=activity_struct{ind}(:,trial_frames_final{target}(trial,:));
        end
    end
end


for ind=1:length(structs)
    for target=1:size(trial_frames_final,2)
        current = structs{1,ind}{1,target};
        for neuron=1:size(current,1)
            pre_act = squeeze(current(neuron,31-pre:31,:));
            post_act = squeeze(current(neuron,32:32+post,:));
            delta{ind}{target}(neuron,:) = mean(post_act,1) - mean(pre_act,1);
        end
        influence{ind}(target,:) = mean(delta{ind}{target}(:,:),2) ./ std(delta{ind}{target}(:,:),0,2);
    end
end


end 


