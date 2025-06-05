%%
figure,
histogram2(imstats_ACLspring.mn,imstats_ACLspring.sd./imstats_ACLspring.mn,...
    0:0.1:32,0:0.005:0.3,'DisplayStyle','tile','Normalization','count')
colormap turbo
set(gca,'ColorScale','log')

%%
destFolder = '/Users/rss367/Desktop/2024bww/Muleshoe/results/ACLSpring'; 
stats = imstats_ACLspring; 

min_max_brightness = 50;
max_avg_brightness = 8;
max_s_m_threshold = 0.25;

null_TF = stats.mx < min_max_brightness;

days_TF = ~null_TF & ~stats.gs;

dark_TF = stats.mn < max_avg_brightness ...
    & stats.sd./stats.mn < max_s_m_threshold ...
    & ~null_TF ...
    & ~days_TF;

dusk_TF = ~null_TF & ~days_TF & ~dark_TF;

null_indices = find(null_TF);
days_indices = find(days_TF);
dark_indices = find(dark_TF);
dusk_indices = find(dusk_TF);


% Create the main category folders under the destination folder.
categories = {'days', 'dusk', 'dark', 'null'};
for i = 1:length(categories)
    catPath = fullfile(destFolder, categories{i});
    if ~exist(catPath, 'dir')
        mkdir(catPath);
    end
end

% Process each category.
copyFilesForCategory(destFolder, stats, 'days', days_indices);
copyFilesForCategory(destFolder, stats, 'dusk', dusk_indices);
copyFilesForCategory(destFolder, stats, 'dark', dark_indices);
copyFilesForCategory(destFolder, stats, 'null', null_indices);

% Create a text file in the destination folder with the provided statistics.
statsFilePath = fullfile(destFolder, 'thresholds.txt');
fid = fopen(statsFilePath, 'w');
if fid == -1
    error('Could not open file %s for writing.', statsFilePath);
end

fprintf(fid, 'min_max_brightness = %g\n', min_max_brightness);
fprintf(fid, 'max_avg_brightness = %g\n', max_avg_brightness);
fprintf(fid, 'max_s/m_threshold = %g\n', max_s_m_threshold);
fclose(fid);


function copyFilesForCategory(destFolder, imstats, categoryName, indices)
    % Loop over each index in the provided list.
    for i = indices
        % Get source folder and file name.
        srcFolder = imstats.files(i).folder;
        fileName = imstats.files(i).name;
        
        % Extract the last part of the source folder as the date folder.
        % Split the path by filesep and get the last non-empty token.
        folderParts = strsplit(srcFolder, filesep);
        % Remove any empty strings (e.g., if path starts with filesep)
        folderParts = folderParts(~cellfun('isempty', folderParts));
        dateFolder = folderParts{end};
        
        % Create destination subfolder: destFolder/categoryName/dateFolder.
        destSubfolder = fullfile(destFolder, categoryName, dateFolder);
        if ~exist(destSubfolder, 'dir')
            mkdir(destSubfolder);
        end
        
        % Build full source and destination file paths.
        srcFile = fullfile(srcFolder, fileName);
        destFile = fullfile(destSubfolder, fileName);
        
        % Copy the file preserving the symlink behavior.
        % Note: This requires a MATLAB version that supports 'slbehavior','preserve'
        try
            copyfile(srcFile, destFile, CopyLinkBehavior="preserve");
        catch ME
            warning('Failed to copy %s to %s: %s', srcFile, destFile, ME.message);
        end
    end
end