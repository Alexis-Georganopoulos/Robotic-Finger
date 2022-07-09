function macroplot(dataset,freq)
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

[N,~] = size(dataset);

idx_range = [1:1:N];
t = (idx_range-1)/freq;
RMS = rms(dataset);

title_space = {'PDC[mV]','Electrode 1[mV]','Electrode 2[mV]',...
               'Electrode 3[mV]','Electrode 4[mV]','Electrode 5[mV]',...
               'Electrode 6[mV]','Electrode 7[mV]','Electrode 8[mV]',...
               'Electrode 9[mV]','Electrode 10[mV]','Electrode 11[mV]',...
               'Electrode 12[mV]','Electrode 13[mV]','Electrode 14[mV]',...
               'Electrode 15[mV]','Electrode 16[mV]','Electrode 17[mV]',...
               'Electrode 18[mV]','Electrode 19[mV]',...
               'X Direction','Y Direction','Z Direction',...
               'All Orientation Components'};

axis_space = {'P_{DC}','X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8',...
              'X_9','X_{10}','X_{11}','X_{12}','X_{13}','X_{14}',...
              'X_{15}','X_{16}','X_{17}','X_{18}','X_{19}','X\in[-1,1]',...
              'Y\in[-1,1]','Z\in[-1,1]','Time[s]'};
figure();          
for i = 1:20
    subplot(5,4,i)
    plot(t,dataset(:,i))
    title(title_space{i})
    xlabel(axis_space{24})
    ylabel(axis_space{i})
    grid on
end

figure();
for i = 21:23
    subplot(2,2,i-20)
    plot(t,dataset(:,i))
    title(title_space{i})
    xlabel(axis_space{24})
    ylabel(axis_space{i})
    grid on
end

subplot(2,2,4)
plot3(dataset(:,21),dataset(:,22),dataset(:,23))
title(title_space{24})
xlabel(axis_space{21})
ylabel(axis_space{22})
zlabel(axis_space{23})
grid on

end

