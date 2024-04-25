
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
addpath('source/2_Raw_Data', 'source/4_Neural_Training')

trial_1 = '2_Raw_Data/biochip_listener_15330_1571242563995.log';
trial_2 = '2_Raw_Data/biochip_listener_15834_1571243162016.log';

T1 = litcount(trial_1);
T2 = litcount(trial_2);

f = 100;

l1 = length(T1);
l2 = length(T2);

t1 = [0:1/f:(l1-1)/f];
t2 = [0:1/f:(l2-1)/f];

figure;
plot(t1,T1(1:end,1))
xlabel('Time[s]')
ylabel('Static pressure[mV]')
figure;
plot(t2,T2(1:end,1))

clear trial_1 trial_2

%% Removing the parts of the data when there is no contact (we NEED pdc)

% for visualisation
f = 100;

l1 = length(T1);
l2 = length(T2);

t1 = [0:1/f:(l1-1)/f];
t2 = [0:1/f:(l2-1)/f];

figure;
plot(t1,T1(1:end,1))
xlabel('Time[s]')
ylabel('Static pressure[mV]')

figure;
plot(t2,T2(1:end,1))

clear f l1 l2

%% Remove T1 extrema(T2 is fine)
%arbitrary cuttoffs for time in the extrema, 0:5s, 62:ends
idx = [find(t1 == 5), find(t1 == 62)];

T1e = T1(idx(1):idx(2),:);
t1e = t1(idx(1):idx(2));

figure;
plot(t1e,T1e(1:end,1))
xlabel('Time[s]')
ylabel('Static pressure[mV]')

figure;
plot(t2,T2(1:end,1))

clear idx t1

%% Lower bounds for T1e and T2

%k1e, k2 arbitrary constants to estimate flatline

k1e = 2440;
k2 = 2460;

L1e = k1e*ones(1,length(t1e));
L2 = k2*ones(1,length(t2));

figure;
plot(t1e,T1e(1:end,1))
hold on
plot(t1e,L1e)

figure;
plot(t2,T2(1:end,1))
hold on
plot(t2,L2)

clear L1e L2

%% Removing initial offset and edge detection

macroplot(T1e,100)
T1s = rem_off_edg(T1e,100,0.05,0.001,1);
macroplot(T1s,100)

[N,~] = size(T1s);
idx_range = [1:1:N];
t = (idx_range-1)/100;
plot(t,T1s(1:end,1))
xlabel('Time[s]')
ylabel('Static pressure[mV]')

macroplot(T2,100)
T2s = rem_off_edg(T2,100,0.05,0.001,1);
macroplot(T2s,100)

clear T1e k1e k2 t1e
%% Removing the first collumn & spliting inputs/outputs for ML regressions

X1 = (T1s(:,2:20))';
X2 = (T2s(:,2:20))';

Y1 = (T1s(:,21:23))';
Y2 = (T2s(:,21:23))';

clear t1s t2s T1s T2s

%The data matrices will now apear like this:

% X:
% |electrodes|
% 
% |__1,1__|__1,2__|__1,3__|__...__|__1,n__| --> n samples of electrode 1
% |__2,1__|__2,2__|__2,3__|__...__|__2,n__| --> n samples of electrode 2
% |__3,1__|__3,2__|__3,3__|__...__|__3,n__| --> n samples of electrode 3
% |__...__|__...__|__...__|__...__|__...__| ...
% |_19,1__|_19,2__|_19,3__|__...__|_19,n__| --> n samples of electrode 19

% Y:
% |z_dir|
% 
% |__1,1__|__1,2__|__1,3__|__...__|__1,n__| --> n samples of x
% |__2,1__|__2,2__|__2,3__|__...__|__2,n__| --> n samples of y
% |__3,1__|__3,2__|__3,3__|__...__|__3,n__| --> n samples of z

%% Save to text file

save('source/4_Neural_Training/X1.txt','X1','-ascii');
save('source/4_Neural_Training/X2.txt','X2','-ascii');
save('source/4_Neural_Training/Y1.txt','Y1','-ascii');
save('source/4_Neural_Training/Y2.txt','Y2','-ascii');

