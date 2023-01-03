clearvars;
%Sudden_change = 0. Gradual change = 1.
%Accuracy = 1         Speed = 0.
global participant_number; 
input = inputdlg('Participant Number?');
participant_number = str2double(input{1});
if mod(participant_number,4) == 0
    participant(participant_number).change = 0;
    participant(participant_number).emphasis = 0;
elseif mod(participant_number,4) == 1
    participant(participant_number).change = 0;
    participant(participant_number).emphasis = 1;
elseif mod(participant_number,4) == 2
    participant(participant_number).change = 1;
    participant(participant_number).emphasis = 0;
else
    participant(participant_number).change = 1;
    participant(participant_number).emphasis = 1;
end   
participant(participant_number).pNo = participant_number;
start;
initial_block;
practice;
after_effects;
fname = sprintf('Data/data%d.mat', participant_number);
save(fname, 'participant');