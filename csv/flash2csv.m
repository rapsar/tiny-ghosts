function [imageData,imageName] = flash2csv(flashFolder,outputname)
%FLASH2CSV reads metadata for flash images and write into a .csv or .txt
% Temperature in Celsius is recorded in the 'MakerNote' field

% parameters
imagePattern = 'DSCF*.JPG';
temperaturePattern = 'tempture';
csvHeader = ["Filename","Year","Month","Day","Hour","Minute","Second","TemperatureC","TemperatureF"];

% Celsius to Fahrenheit
C2F = @(x) 1.8*x+32;

% list files
if flashFolder(end) ~= '/'
    flashFolder(end+1) = '/';
end
filePattern = strcat(flashFolder,imagePattern);
files = dir(filePattern);
nFiles = length(files);

% initialize
imageName = strings(nFiles,1);
imageData = NaN(nFiles,8);

for i=1:nFiles

    fileName = files(i).name;
    filePath = strcat(flashFolder,fileName);

    % metadata
    info = imfinfo(filePath);

    % date-time
    dt = info.DateTime;
    yMdhms = datetime(dt,'Format','yyyy:MM:dd HH:mm:ss');

    % temperature
    MakerNote = info.MakerNote;
    k = strfind(MakerNote,temperaturePattern);
    TC = str2double(MakerNote(k+9:k+10));
    TF = round(C2F(TC));
    
    % fill arrays
    imageName(i,1) = fileName;
    imageData(i,:) = [year(yMdhms), month(yMdhms), day(yMdhms),...
                        hour(yMdhms), minute(yMdhms), second(yMdhms),... 
                        TC, TF];
end

% sort by ascending date
[imageData,idx] = sortrows(imageData,[1 2 3 4 5 6]);
imageName = imageName(idx);

% compile csv data
csvData = [csvHeader ; imageName , imageData];
% write csv
writematrix(csvData,outputname);

end

