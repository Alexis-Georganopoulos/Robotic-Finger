function [newdataset] = rem_off_edg(dataset,sfreq,offtol,edgetol,lp)
%offtol and edgetol are percentages of the rms between [0,1]
% The format of the data structure is as follows:
% (the numbers in parenthesis are the column indices for matlab,
% likewise the table below is the raw structure)
%     
% |pdc:(1)|electrode_1(2)... electrode_19(20)| z_dir_x(21)... z_dir_z(23)|
% |__1,1__|_______________1,2:20_____________|_________1,21:23___________|
% |__2,1__|_______________2,2:20_____________|_________2,21:23___________|
% |__3,1__|_______________3,2:20_____________|_________3,21:23___________|
%                                   ...
% |__n,1__|_______________n,2:20_____________|_________n,21:23___________|
%
%
% The row indices are range from 1 to the number of samples taken(n)
%
% The data for this project was sampled at around 100[Hz]


[N,M] = size(dataset);
newdataset = dataset;
pdc = dataset(:,1);
%Offset removal
pdcdt = diff(pdc)*sfreq;
pdcdt = lowpass(pdcdt,lp,sfreq);
RMSO = offtol*rms(pdcdt);
temp = find(pdcdt >= RMSO);
idxO = temp(1);

Offsets = mean(dataset(1:idxO,1:21));

for i = 1:M-3
    newdataset(:,i) = newdataset(:,i)-Offsets(i);
end

%Edge removal
RMSE = edgetol*rms(pdc);
P = polyfit([1,N],[pdc(1),pdc(N)],1);
Y = polyval(P,[1:1:N]);

temp_pdc = pdc - Y';
idxe = find(temp_pdc > RMSE);

newdataset = newdataset(idxe,:);
    

end

