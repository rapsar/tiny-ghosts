function [] = processHighKurtosisImages(imstats)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

kurtosis_minimum = 10;
avg_brightness_maximum = 10;
max_brightness_minimum = 50;

%% find isolated high kurtosis images
% find high kurtosis
clusters = (imstats.ks > kurtosis_minimum);

% compute size of high kurtosis 1D clusters
rp = regionprops(clusters,imstats.ks,'Area','PixelIdxList','PixelValues');

% find isolated high kurtosis peaks
pk = (vertcat(rp.Area) == 1);
isolated_high_kurt_idx = vertcat(rp(pk).PixelIdxList);

% find high kurtosis image with bright enough pixel but low enough average
high_max = imstats.mx(isolated_high_kurt_idx)' > max_brightness_minimum;
low_mean = imstats.mn(isolated_high_kurt_idx)' < avg_brightness_maximum;
image_candidates_idx = isolated_high_kurt_idx(low_mean & high_max);

%% calculate number and area of bright blobs
j = 1;
for i = image_candidates_idx'
    filename = fullfile(imstats.files(i).folder, imstats.files(i).name);
    im = imread(filename);
    im = im(1:1280,:,1);
    bw = im > max_brightness_minimum;
    rp2{i} = regionprops(bw,'Area');
    area{i} = vertcat(rp2{i}.Area);
    out(j,:) = [numel(area{i}) max(area{i})];
    j = j+1;
end

%% keep images with few blobs and small enough
image_indices = image_candidates_idx(out(:,1) < 4 & out(:,2) < 300);

%%
fileIndices = image_indices;
%inputFolder = '/Users/rss367/Desktop/2024bww/Muleshoe/results/HotSpringsStoneCabin/high_kurtosis';
destinationFolder = '/Users/rss367/Desktop/2024bww/Muleshoe/results/WillowSprings/high_kurtosis_small_area';

if ~exist(destinationFolder, 'dir')
    mkdir(destinationFolder);
end
% Loop through each filename in the list
for i = fileIndices'
    %sourceFile = fullfile(inputFolder, d(i).name);
    sourceFile = fullfile(imstats.files(i).folder, imstats.files(i).name);
    destinationFile = fullfile(destinationFolder, imstats.files(i).name);

    % Check if the source file exists before copying
    if exist(sourceFile, 'file')
        copyfile(sourceFile, destinationFile);
        fprintf('Copied: %s -> %s\n', sourceFile, destinationFile);
    else
        fprintf('File not found: %s\n', sourceFile);
    end
end
end

