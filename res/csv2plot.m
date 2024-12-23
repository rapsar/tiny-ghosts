function [out] = csv2plot(csvPath,minuteAverage)
%CSV2PLOT Plot data in csv files
% minuteAverage is number of minutes to for binning 

M = readmatrix(csvPath);

% pulls data
timeData = datetime(M(:,2:7));
tempData = M(:,8);

edges = timeData(1):duration(0,minuteAverage,0):timeData(end);

figure,

histogram(timeData,edges)
xlabel('time')
ylabel('Number of Flash Pictures')
xlim([timeData(1) timeData(end)])

yyaxis right
hold on
plot(timeData,tempData,'r.-')
ylabel('Temperature (C)')

title(['Flash Trends, ' datestr(timeData(1))])

%% return
out.date = timeData(1);
out.minuteAverage = minuteAverage;
out.edges = edges;
out.n = histcounts(timeData,edges);


end

