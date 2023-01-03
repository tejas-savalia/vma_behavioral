pno = 1;
xCenter = 960;
yCenter = 540;
for pno = 1000:1060
    fload = sprintf('C:\\Users\\Tejas\\Documents\\Research\\Lab-Codes\\Psychtoolbox\\Visuomotor_Adaptation_Tablet\\Data\\data%d.mat', pno);
    load(fload);
    squares = zeros(64, 1);
    new_dir = sprintf('python_scripts\\data\\data%d\\squares', pno);
    mkdir (new_dir);

    for i = 1:10
        squares = participant(pno).practice.block(i).squares;  
        %squareX = 300*cos(squares);
        %squareY = 300*sin(squares);
        fname = sprintf('python_scripts\\data\\data%d\\squares\\squares%d', pno, i);
        save(fname, 'squares');
    %    fname = sprintf('python_scripts\\data\\pilot\\pilot_%d\\squares\\coordinates\\squares%d', pno, i);
    %    save(fname, 'squareX', 'squareY');
    end

    for i = 1:1
        squares = participant(pno).ib.block(i).squares;          
        %squareX = 300*cos(squares);
        %squareY = 300*sin(squares);
        fname = sprintf('python_scripts\\data\\data%d\\squares\\squares0', pno);
        save(fname, 'squares');
        %fname = sprintf('python_scripts\\data\\pilot\\pilot_%d\\squares\\coordinates\\squares0', pno);
        %save(fname, 'squareX', 'squareY');

    end
    for i = 1:1
        squares = participant(pno).ae.block(i).squares;  
        %squareX = 300*cos(squares);
        %squareY = 300*sin(squares);
        fname = sprintf('python_scripts\\data\\data%d\\squares\\squares11', pno);
        save(fname, 'squares');
        %fname = sprintf('python_scripts\\data\\pilot\\pilot_%d\\squares\\coordinates\\squares11', pno);
        %save(fname, 'squareX', 'squareY');

    end

end
    %randomSquareTheta = participant(pno).practice.block(block).squares(i)   ;
    %randomSquareXpos = 300*cos(randomSquareTheta) + xCenter;
    %randomSquareYpos = 300*sin(randomSquareTheta) + yCenter;

