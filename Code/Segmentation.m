%% Setup
% Define the directories and file names of the NIfTI MRI data.
fileDir= "/shared/mrfil-data/VesselTracking/jp/data/Raw-Data/Restored_img_4seg";

% Combine directory and file name for full paths.
fullFilePath_566 = fullfile(fileDir, "anat-angio_acq-TOF_axial_restore_566_aligned.nii.gz");
fullFilePath_1758 = fullfile(fileDir, "anat-angio_acq-TOF_axial_restore_1758_aligned.nii.gz");
fullFilePath_1539 = fullfile(fileDir, "anat-angio_acq-TOF_axial_restore_1539.nii.gz");

%% Load Image Data
% Load the MRI data from NIfTI files using the 'niftiread' function.
img_1539 = niftiread(fullFilePath_1539);
img_566 = niftiread(fullFilePath_566);
img_1758 = niftiread(fullFilePath_1758);

%% Plot Cumulative Distribution Functions (CDFs)
figure;
hold on;

% CDF for Image 1539
cdfplot(img_1539(:));
% CDF for Image 566
cdfplot(img_566(:));
% CDF for Image 1758
cdfplot(img_1758(:));

% Customize plot
legend({'Image 1539', 'Image 566', 'Image 1758'}, 'Location', 'best');
title('Cumulative Distribution Functions of Original Intensities');
xlabel('Intensity');
ylabel('Cumulative Probability');
grid on;

hold off;

% Interpretation:
% - If the curves are close together, the intensity distributions are similar.
% - If the curves are far apart, the distributions are different.


%% Step 1: Define the Center Point & ROI Selection
centerX = 239;
centerY = 206;
centerZ = 120;

range_xy = 50;
range_z = 80;

% Define the ranges based on the center and desired range
xRange = centerX-range_xy:centerX+range_xy;  % X-direction
yRange = centerY-range_xy:centerY+range_xy;  % Y-direction
zRange = centerZ-range_z:centerZ+range_z;    % Z-direction

% Extract the ROI centered around the specified ranges
img_ROI = img_1539(yRange, xRange, zRange);

% Visualize the central slice of the extracted ROI (optional)
centralSlice = round(size(img_ROI, 3) / 2); % The middle slice in the Z direction
figure;
imshow(img_ROI(:,:,centralSlice), []);
title('Central Slice of the ROI Around the Specified Center Point');

%% Step 2: Find the Maximum Intensity Value within the ROI
[maxIntensity, linearIndex] = max(img_ROI(:));

% Convert the linear index to subscripts to get the (x, y, z) coordinates in ROI
[coordY, coordX, coordZ] = ind2sub(size(img_ROI), linearIndex);

% Display the results within the ROI
fprintf('Max Intensity: %f\n', maxIntensity);
fprintf('Coordinates in ROI (Y, X, Z): (%d, %d, %d)\n', coordY, coordX, coordZ);

% Convert the coordinates to the original image space
originalCoordX = xRange(coordX);
originalCoordY = yRange(coordY);
originalCoordZ = zRange(coordZ);

% Display the adjusted coordinates in the original image
fprintf('Coordinates in Original Image (Y, X, Z): (%d, %d, %d)\n', originalCoordY, originalCoordX, originalCoordZ);

%% Step 3: Run the Region Growing Algorithm
seeds = [coordY, coordX, coordZ];  % Use the coordinates found earlier as seeds
initialThreshold = 425;  % Starting threshold
minThreshold = 50;  % Minimum threshold to allow finer control as the region grows
patchSize = 3;  % Define the patch size (2, 3, or 4)

% Run the region growing algorithm
regionMask = adaptiveRegionGrowing(img_ROI, seeds, initialThreshold, minThreshold, patchSize);

%% Step 4: Visualize the Result in 2D
figure;
imshowpair(img_ROI(:,:,centralSlice), regionMask(:,:,centralSlice), 'blend');
title('Region Growing Segmentation Result');

%% Step 5: Visualize the Segmented Mask in 3D
% Use isosurface to visualize the 3D structure of the segmented mask
figure;
p = patch(isosurface(regionMask, 0.5));
isonormals(regionMask, p);
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1]);
view(3);
axis tight;
camlight;
lighting gouraud;
title('3D Visualization of the Region Growing Segmentation');

%% Adaptive Region Growing Function

%Psuedo Code for Region Growing Algorithm
%seeds location = input()  ## take object seed to be segmented as input
%each region mean  = seed value
%set error as similarity measure for regions
%while there is unallocated pixels 
%    for each pixel in each region
%        if  unallocated neighbors of (2x2 , 3x3 or 4x4) patches  within a specified erros from region mean
%            then add to region and recalculate region mean
%        else
%            mark as visited and dont add to region

function regionMask = adaptiveRegionGrowing(img, seeds, initialThreshold, minThreshold, patchSize)
    % Initialize the region mask
    regionMask = false(size(img));
    
    % Initialize a visited mask to keep track of visited pixels
    visited = false(size(img));
    
    % Initialize the region mean with the seed values
    regionMeans = img(seeds(1), seeds(2), seeds(3));
    
    % Create a list of region pixels to expand
    regionPixels = seeds;
    
    % Define the neighborhood patch based on patchSize
    switch patchSize
        case 2
            neighborhood = [-1, -1, 0; -1, 0, 0; -1, 1, 0; 0, -1, 0; 0, 1, 0; 1, -1, 0; 1, 0, 0; 1, 1, 0];
        case 3
            [x, y, z] = meshgrid(-1:1, -1:1, -1:1);
            neighborhood = [x(:), y(:), z(:)];
        case 4
            [x, y, z] = meshgrid(-2:2, -2:2, -2:2);
            neighborhood = [x(:), y(:), z(:)];
        otherwise
            error('Invalid patch size. Choose 2, 3, or 4.');
    end
    
    % Adaptive region growing loop
    while ~isempty(regionPixels)
        % Get the first pixel to expand
        currentPixel = regionPixels(1, :);
        regionPixels(1, :) = [];  % Remove it from the list
        
        % Mark the current pixel as visited
        visited(currentPixel(1), currentPixel(2), currentPixel(3)) = true;
        
        % Adjust the threshold adaptively based on the region size or distance from the seed
        adaptiveThreshold = initialThreshold - (initialThreshold - minThreshold) * (nnz(regionMask) / numel(regionMask));
        adaptiveThreshold = max(adaptiveThreshold, minThreshold); % Ensure it doesn't go below minThreshold
        
        % Check the neighbors
        for i = 1:size(neighborhood, 1)
            neighbor = currentPixel + neighborhood(i, :);
            
            % Ensure the neighbor is within image bounds
            if all(neighbor >= 1) && neighbor(1) <= size(img, 1) && neighbor(2) <= size(img, 2) && neighbor(3) <= size(img, 3)
                
                % If the neighbor has not been visited
                if ~visited(neighbor(1), neighbor(2), neighbor(3))
                    % Calculate the difference between the neighbor and the region mean
                    intensityDiff = abs(img(neighbor(1), neighbor(2), neighbor(3)) - mean(regionMeans));
                    
                    % If the difference is within the adaptive threshold, add the neighbor to the region
                    if intensityDiff <= adaptiveThreshold
                        regionMask(neighbor(1), neighbor(2), neighbor(3)) = true;
                        regionPixels = [regionPixels; neighbor];  % Add the neighbor to the list to expand
                        regionMeans = [regionMeans; img(neighbor(1), neighbor(2), neighbor(3))];  % Update the region mean
                    end
                    
                    % Mark this neighbor as visited
                    visited(neighbor(1), neighbor(2), neighbor(3)) = true;
                end
            end
        end
    end
end

