
sca;
close all;
%clearvars;

PsychDefaultSetup(2);

screens = Screen('Screens');
screenNumber = max(screens);
%screenNumber = 1;
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);
grey = white/2;

[window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey);
[screenXpixels, screenYpixels] = Screen('WindowSize', window);
[xCenter, yCenter] = RectCenter(windowRect);

ifi = Screen('GetFlipInterval', window);
numSecs = 10;
numFrames = round(numSecs/ifi);
vb1 = Screen('Flip', window);
SetMouse(xCenter, yCenter, window);

dotColor = [1, 0, 0];
dotSizePix = 20; 
HideCursor();

Screen('DrawDots', window, [xCenter, yCenter], dotSizePix, dotColor, [], 2);
Screen('Flip', window);    
ibXs = {};
ibYs = {};

baseRect = [0 0 100 100];
maxDiameter = max(baseRect)*1.05;
numRects = 4;

%Screen('Flip', window);

numBlocks = 10;
numTrials = numRects*16;

%Gradual vs Sudden change
if participant(participant_number).change == 0
    participant(participant_number).rotateBy(1:numBlocks) = 90;
elseif participant(participant_number).change == 1
    participant(participant_number).rotateBy = [10:90/(numBlocks-1):90 90];
end



totalScore = 0;
times = zeros(numBlocks, numTrials);
ibsquares = zeros(numBlocks, numTrials);
initial_time = zeros(numBlocks, numTrials);
for block = 1:numBlocks
    
    squareTheta = repelem([pi/4, 3*pi/4, 5*pi/4, 7*pi/4], numTrials/16);
    %i = randperm(length(squareTheta));
    randomSquareThetaVec = [squareTheta(:, randperm(length(squareTheta))) squareTheta(:, randperm(length(squareTheta))) squareTheta(:, randperm(length(squareTheta))) squareTheta(:, randperm(length(squareTheta)))];
    %ibsquares(block, :) = randomSquareThetaVec;
    participant(participant_number).practice.block(block).squares = randomSquareThetaVec;
    %Code for inter block interval
    blockScore = 0;

    Screen('FillRect', window, [0.5, 0.5, 0.5]);        
    Screen('TextSize', window, 30);
    DrawFormattedText(window, 'Ready?', xCenter - 100, yCenter, [1 0 0]);
    DrawFormattedText(window, 'Press any key to Continue', xCenter-350, yCenter + 100, [1 0 0]);
    Screen('Flip', window);
    KbStrokeWait;
    
    for trial = 1:numTrials
        newXs = [];
        newYs = [];
        randomSquareTheta = randomSquareThetaVec(trial);
        randomSquareXpos = 300*cos(randomSquareTheta) + xCenter;
        randomSquareYpos = 300*sin(randomSquareTheta) + yCenter;

        randomSquareColor = [1, 0, 0];
        changeSquareColor = [0, 1, 0];
        randomSquare = CenterRectOnPointd(baseRect, randomSquareXpos, randomSquareYpos);

        %Code for inter-trial-interval
        %Auditary cue. 
        %Screen('FillRect', window, randomSquareColor, randomSquare);
        Screen('TextSize', window, 30);
        Screen('FillRect', window, [0, 0.5, 0.5])
        %DrawFormattedText(window, num2str(blockScore), screenXpixels*0.80 ,screenYpixels * 0.15, [1 0 0]);
        Screen('DrawDots', window, [xCenter, yCenter], dotSizePix, dotColor, [], 2);        
        
        Screen('Flip', window);    

        rand_interval = 1 + rand(1, 1)*(3 - 1);
        pause(rand_interval);
        
        
        tic;
        first_flag = true;
        while true
        
            Screen('FillRect', window, [0, 0.5, 0.5]);
            
            [x, y, buttons] = GetMouse(window);
            %theta = atan(y-yCenter/x-xCenter);
            %r = (x-xCenter)/cos(theta);
            %theta = acos((x - xCenter)/r);

            % Draw the rect to the screen
            %Screen('FillRect', window, allColors, allRects);
            Screen('FillOval', window, randomSquareColor, randomSquare, maxDiameter);
            Screen('TextSize', window, 30);
            %DrawFormattedText(window, num2str(blockScore), screenXpixels*0.80 ,screenYpixels * 0.15, [1 0 0]);



            if buttons(1)
                if first_flag
                    participant(participant_number).practice.block(block).trial(trial).initial_time = toc;
                    xCenter_ = x;
                    yCenter_ = y;
                    first_flag = false;
                end
                HideCursor();
                %SetMouse(x + r*cos(theta+pi/4), y + r*sin(theta+pi/4), window);
                newX = (x-xCenter_)*cosd(participant(participant_number).rotateBy(block)) + (y-yCenter_)*sind(participant(participant_number).rotateBy(block));
                newY = -(x-xCenter_)*sind(participant(participant_number).rotateBy(block)) + (y-yCenter_)*cosd(participant(participant_number).rotateBy(block));
                newXs = [newXs newX];
                newYs = [newYs newY];
                
                inside = IsInRect(newX+xCenter, newY+yCenter, randomSquare);
                
                if inside
                    %printf('here');
                    %randomSquareTheta = randomSquareThetaVec(trial+1);
                    %randomSquareXpos = 300*cos(randomSquareTheta) + xCenter;
                    %randomSquareYpos = 300*sin(randomSquareTheta) + yCenter;

                    %randomSquareColor = [1, 0, 0];
                    %randomSquare = CenterRectOnPointd(baseRect, randomSquareXpos, randomSquareYpos);


                    %blockScore = blockScore + 1;
                    participant(participant_number).practice.block(block).trial(trial).movementTime = toc; 
                    break;
                end


                Screen('DrawDots', window, [xCenter+newX, yCenter+newY], dotSizePix, dotColor, [], 2);        
            end
            if ~buttons(1)

        
        %display score
        
                SetMouse(xCenter, yCenter, window);
                Screen('DrawDots', window, [xCenter, yCenter], dotSizePix, dotColor, [], 2); 
                if ~first_flag
                    first_flag = true;
                    continue;
                end
            end
            Screen('Flip', window);
        end
        
        Screen('Flip', window);
        if participant(participant_number).emphasis
            blockScore = blockScore + 1000/RMSE(newXs, newYs, xCenter, yCenter, randomSquareXpos, randomSquareYpos);
        elseif ~participant(participant_number).emphasis
            blockScore = blockScore + 1/participant(participant_number).practice.block(block).trial(trial).initial_time;
        end
       participant(participant_number).practice.block(block).trial(trial).yTrajectory = newYs;
       participant(participant_number).practice.block(block).trial(trial).xTrajectory = newXs;    
    end
    %display leaderboard.

    
    Screen('FillRect', window, [0.5, 0.5, 0.5]);
    
    participant(participant_number).practice.blockScore(block) = blockScore;
    Screen('TextSize', window, 30);
    %DrawFormattedText(window, num2str(totalScore), xCenter, yCenter, [1 0 0]);
    %sb = load('scoreboard.mat');
    %sortedboard = sort([round(sb.scoreboard(block, :)) round(blockScore)]);
    scoreboard = normrnd(round(blockScore), round(blockScore)/5, [4, 1]);
    sortedBoard = sort([round(blockScore), transpose(round(scoreboard))]);
    DrawFormattedText(window, sprintf('Scoreboard: \n'), xCenter - 150, yCenter - 250, [1 0 0]);
    for sortedScore = 1:5
        if sortedBoard(sortedScore) == round(blockScore)
            DrawFormattedText(window, sprintf('%d\n', sortedBoard(sortedScore)), xCenter - 100, yCenter - (250-(sortedScore*40)), [0 1 0]);
        else
            DrawFormattedText(window, sprintf('%d\n', sortedBoard(sortedScore)), xCenter - 100, yCenter - (250-(sortedScore*40)), [1 0 0]);
        end
    end
    
    if participant(participant_number).emphasis
        DrawFormattedText(window, sprintf('You can always increase your score by doing it more accurately!'), xCenter-450, yCenter + 50, [1 0 0]);
    elseif ~participant(participant_number).emphasis
        DrawFormattedText(window, sprintf('You can always increase your score by doing it faster!'), xCenter-450, yCenter + 50, [1 0 0]);
    end
    DrawFormattedText(window, sprintf('%d more to go!', numBlocks-block+1), xCenter-450, yCenter , [1 0 0]);    
    DrawFormattedText(window, 'Take a Break! Press any key to Continue', xCenter-450, yCenter + 100, [1 0 0]);
    
    Screen('Flip', window);
    KbStrokeWait;
    
 end
%SetMouse(400, 400, window)
%KbStrokeWait;

% Flip to the screen
Screen('Flip', window);
%KbStrokeWait;
    
sca;

%TX = cell2table(ibXs, 'VariableNames', {'block' 'trial'});
%TY = cell2table(ibYs, 'VariableNames', {'block' 'trial'}); 
% Write the table to a CSV file
%writetable(TX,'initialX.csv');
%writetable(TY,'initialY.csv');

%scatter(Xs{1}, -Ys{1});