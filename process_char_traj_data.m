% Load the Character Trajectories dataset from the .mat file
load('mixoutALL_shifted.mat');

% Access the 'mixout' cell array containing character samples
character_samples = mixout;

% Initialize empty matrices to store data
x_values = [];
y_values = [];
force_values = [];
time_values = [];
labels = [];

% Loop through each character sample
for i = 1:length(character_samples)
    % Extract x, y, force, and time values for the current character sample
    trajectory = character_samples{i};
    x = trajectory(1, :);
    y = trajectory(2, :);
    force = trajectory(3, :);
    
    % Calculate time values based on the sampling rate of 200Hz
    time = (1:length(x)) / 200;
    
    % Concatenate the values into the respective matrices
    x_values = [x_values, x];
    y_values = [y_values, y];
    force_values = [force_values, force];
    time_values = [time_values, time];
    
    % Add labels corresponding to the character
    labels = [labels, repmat(i, 1, length(x))];
end

% Concatenate the matrices to form the final matrix where columns represent (x, y, force, time)
character_matrix = [x_values; y_values; force_values; time_values; labels];

% Transpose the matrix so that columns represent (x, y, force, time, label)
character_matrix = character_matrix';

% Save the data to a CSV file
csvwrite('character_trajectories_matrix.csv', character_matrix);
disp('Data saved to character_trajectories_matrix.csv');
