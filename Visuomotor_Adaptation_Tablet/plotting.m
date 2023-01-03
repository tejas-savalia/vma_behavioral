%To save trajectories from practice trials, run this block.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for pno = 1000:1060
%{    
    for i = 1:10
        fload = sprintf('Data\\data%d.mat', pno);
        load(fload);
        [x, y] = trial_traj(participant, pno, i);
        new_dir = sprintf('python_scripts\\data\\data%d\\actual_trajectories', pno);
        mkdir (new_dir);
        fname = sprintf('python_scripts\\data\\data%d\\actual_trajectories\\trajectories%d', pno, i);
        save(fname, 'x', 'y');
        
        [idealXs, idealYs] = ideal_trajectories(participant, pno, i);
        new_dir = sprintf('python_scripts\\data\\data%d\\ideal_trajectories', pno);
        mkdir (new_dir);
        fname = sprintf('python_scripts\\data\\data%d\\ideal_trajectories\\trajectories%d', pno, i);
        save(fname, 'idealXs', 'idealYs');
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for pno = 1044:1048

    for i = 1:1
        fload = sprintf('Data\\data%d.mat', pno);
        load(fload);
        [x, y] = trial_traj_ae(participant, pno, i);
        new_dir = sprintf('python_scripts\\data\\data%d\\actual_trajectories', pno);
        mkdir (new_dir);
        fname = sprintf('python_scripts\\data\\data%d\\actual_trajectories\\trajectories11', pno);
        save(fname, 'x', 'y');
        
        [idealXs, idealYs] = ideal_trajectories_ae(participant, pno, i);
        new_dir = sprintf('python_scripts\\data\\data%d\\ideal_trajectories', pno);
        mkdir (new_dir);
        fname = sprintf('python_scripts\\data\\data%d\\ideal_trajectories\\trajectories11', pno);
        save(fname, 'idealXs', 'idealYs');
    end
%end
%}

%for pno = 1044:1048
    %if pno == 22 || pno == 25 || pno == 27 || pno == 43
    %    continue;
    %end
    for i = 1:1
        fload = sprintf('Data\\data%d.mat', pno);
        load(fload);
        [x, y] = trial_traj_ib(participant, pno, i);
        new_dir = sprintf('python_scripts\\data\\data%d\\actual_trajectories', pno);
        mkdir (new_dir);
        fname = sprintf('python_scripts\\data\\data%d\\actual_trajectories\\trajectories0', pno);
        save(fname, 'x', 'y');
        
        [idealXs, idealYs] = ideal_trajectories_ib(participant, pno, i);
        new_dir = sprintf('python_scripts\\data\\data%d\\ideal_trajectories', pno);
        mkdir (new_dir);
        fname = sprintf('python_scripts\\data\\data%d\\ideal_trajectories\\trajectories0', pno);
        save(fname, 'idealXs', 'idealYs');
    end
end
