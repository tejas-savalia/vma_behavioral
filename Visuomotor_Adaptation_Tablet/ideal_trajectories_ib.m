function [idealXs, idealYs] = ideal_trajectories_ib(participant, pno, block)
xCenter = 960;
yCenter = 540;
[x, y] = trial_traj_ib(participant, pno, block);

for i = 1:64
    %randomSquareTheta = participant(pno).practice.block(block).squares(i)   ;
    %randomSquareXpos = 300*cos(randomSquareTheta) + xCenter;
    %randomSquareYpos = 300*sin(randomSquareTheta) + yCenter;

    %[theta, rho] = cart2pol(randomSquareXpos-xCenter, randomSquareYpos - yCenter);
    %theta = wrapTo2Pi(theta);

    samples = length(x{i});
    randomSquareXpos = x{i}(samples);
    randomSquareYpos = y{i}(samples);

    xratios = x{i}(1:samples)/sum(x{i}(1:samples));
    xratios = cumsum(xratios);
    yratios = y{i}(1:samples)/sum(y{i}(1:samples));
    yratios = cumsum(yratios);
    %ideal trajectory point spacings proportional to actual trajectory
    %point spacings. 
%    for j = 1:samples
%        A = [x{i}(j), y{i}(j)];
%        B = [randomSquareXpos, randomSquareYpos];
%        ideals = (dot(A, B)/norm(B))*(B/norm(B));
%    end
    idealXs{i} = (randomSquareXpos)*xratios;
    idealYs{i} = (randomSquareYpos)*yratios;
 
end
end