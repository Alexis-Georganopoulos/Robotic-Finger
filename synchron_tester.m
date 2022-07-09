fh = 120;
fl = 100;

th = [0:1/fh:12];
tl = [0:1/fl:12];
ffh = 0.1*ones(1,length(th));
ffl = zeros(1,length(tl));

plot(th,ffh,'--o',tl,ffl,'--o')