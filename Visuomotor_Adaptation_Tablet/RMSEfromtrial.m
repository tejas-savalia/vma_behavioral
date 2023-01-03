function[rmse] = RMSEfromtrial(participant, block)
xCenter = 960;
yCenter = 540;
[x, y] = trial_traj(participant);
rmse = zeros(64, 1);
for i = 1:64    
    randomSquareTheta = participant(1).practice.block(block).squares(i)   ;
    randomSquareXpos = 300*cos(randomSquareTheta) + xCenter;
    randomSquareYpos = 300*sin(randomSquareTheta) + yCenter;
    rmse(i,1) = RMSE(x{i}, y{i}, xCenter, yCenter, randomSquareXpos, randomSquareYpos);
end
end
