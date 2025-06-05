%function imstats = calculateImagesStats(inputFolder)
% calculateImagesStats Calculate image statistics (mean, std, kurtosis, max)
% Processes all *DSCF*.JPG files in inputFolder.
% If no files are found, checks subfolders and processes all files inside.

% Using a script rather than a function for flexibility, and in case it
% breaks before completing
    
    inputFolder = '/Users/rss367/Desktop/2024bww/Muleshoe/data/ACLSpring';

    imstats_ACLspring.inputFolder = inputFolder;
    allFiles = [];  % Initialize empty structure array

    % List files in inputFolder
    files = dir(fullfile(inputFolder, '*DSCF*.JPG'));
    if isempty(files)
        % If no files, look for subfolders
        subfolders = dir(inputFolder);
        subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name}, '.'));

        for k = 1:length(subfolders)
            subfolderPath = fullfile(inputFolder, subfolders(k).name);
            subfolderFiles = dir(fullfile(subfolderPath, '*DSCF*.JPG'));

            % Append found files to the list
            if ~isempty(subfolderFiles)
                for j = 1:length(subfolderFiles)
                    subfolderFiles(j).folder = subfolderPath; % Correct folder path
                end
                allFiles = [allFiles; subfolderFiles]; %#ok<AGROW>
            end
        end
    else
        allFiles = files;
    end

    % Check if there are files to process
    nFiles = length(allFiles);
    if nFiles == 0
        fprintf('No images found in %s or its subfolders.\n', inputFolder);
        imstats_ACLspring.files = [];
        return;
    end

    imstats_ACLspring.files = allFiles;
    fprintf('Started processing %d files...\n', nFiles);

    % Initialize stats arrays
    imstats_ACLspring.gs = zeros(1, nFiles);
    imstats_ACLspring.mn = zeros(1, nFiles);
    imstats_ACLspring.sd = zeros(1, nFiles);
    % imstats_upperwildcat.ks = zeros(1, nFiles);
    imstats_ACLspring.mx = zeros(1, nFiles);

    tic;
    w = waitbar(0, 'Processing images...');
    
    for i = 1:nFiles % use parfor?
        filename = fullfile(allFiles(i).folder, allFiles(i).name);

        im = imread(filename);

        im = im(1:1280, :, :);  % remove bottom banner

        imr = im(:,:,1);        % red channel
        img = im(:,:,2);        % green channel

        % check if image is rgb or grayscale
        imstats_ACLspring.gs(i) = all(imr(:) == img(:));

        imr = double(imr(:));

        imstats_ACLspring.mn(i) = mean(imr);
        imstats_ACLspring.sd(i) = std(imr);
        % imstats_upperwildcat.ks(i) = kurtosis(imr);
        imstats_ACLspring.mx(i) = max(imr);

        waitbar(i / nFiles, w);
    end

    close(w);
    fprintf('Finished processing %d images.\n', nFiles);
    toc;
%end

