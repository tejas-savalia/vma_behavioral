function [rmse] = RMSE(xTrajectory, yTrajectory, xCenterPos, yCenterPos, xTargetPos, yTargetPos)

samples = length(xTrajectory);
idealXs = linspace(0, xTargetPos - xCenterPos, samples);
idealYs = linspace(0, yTargetPos - yCenterPos, samples);

rmse = sqrt(sum((xTrajectory - idealXs).^2 + (yTrajectory - idealYs).^2)/samples);

end
