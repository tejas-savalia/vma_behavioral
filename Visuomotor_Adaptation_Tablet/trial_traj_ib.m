%Function to obtain trial trajectories from xTrajectory and yTrajectory
%variables
function[xtrial_traj, ytrial_traj] = trial_traj_ib(participant, pno, block)

%for i = 1:64
%    xtrial_traj{i} = participant(pno).ib.block(block).trial(i).xTrajectory;
%    ytrial_traj{i} = participant(pno).ib.block(block).trial(i).yTrajectory;
%end
xtrial_traj{1} = participant(pno).ib.block(block).trial(1).xTrajectory;
ytrial_traj{1} = participant(pno).ib.block(block).trial(1).yTrajectory;

for i = 1:63
    last_cutoff = length(participant(pno).ib.block(block).trial(i+1).xTrajectory);
    first_cutoff = length(participant(pno).ib.block(block).trial(i).xTrajectory);
    xtrial_traj{i+1} = participant(pno).ib.block(block).trial(64).xTrajectory(1+first_cutoff:last_cutoff);
    ytrial_traj{i+1} = participant(pno).ib.block(block).trial(64).yTrajectory(1+first_cutoff:last_cutoff);
    
end


%Calculate xCenter yCenter from the screeen first
