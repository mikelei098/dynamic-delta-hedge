clear;
set(0, 'defaultTextFontSize',24);
% Import data from text file
% Setup the Import Options
opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double"];
opts = setvaropts(opts, 1, "InputFormat", "yyyy-MM-dd");
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
M = readtable("AAPL.csv", opts);
        
% Clear temporary variables
clear opts

% Setup the Import Options
opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double"];
opts = setvaropts(opts, 1, "InputFormat", "MM/dd/yyyy");
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
D = readtable("AAPL_D.csv", opts);


% Clear temporary variables
clear opts

%% 
R = M.Close(end)/M.Open(1);
R = R^(1/60)-1; % monthly return
V = 0;
for i = 2:60
    r = (M.Close(i)-M.Close(i-1))/M.Close(i-1);
    V = V +(r-R)^2;
end

V  = sqrt(V/59); % volatility


%% Dynamic delta strategy
call = NaN(60,1);
Kc = NaN(60,1);
del = NaN(90,57);
C = NaN(90,57);
SH = NaN(90,57);
DELTA = NaN(60,1);
exe = boolean(zeros(60,1));
dyn = NaN(60,1);
con1 = NaN(60,1);
con2 = NaN(60,1);
QR = (1+R)^3;  % quartly return
QV = sqrt(3)*V; % quartly volatility
rf = (1+0.02)^(1/4)-1; % quartly risk-free rate
for i = 1:57
    d1 = M.Date(i); % start date
    d2 = M.Date(i+3); % mature date
    row1 = find(D.Date == d1);
    row2 = find(D.Date == d2);
    t = row2-row1; % duration
    T =([0:-1:-t]'+t)/(t);  % time values for black scholes model
    T(end,1) = 0.000001;  % correct the last entry to be non-zero
    S0 = D.Close(row1);
    Kc(i+3) = S0*QR;
    del(1,i) = normcdf((log(D.Close(row1)/Kc(i+3))+(rf+QV^2/2)*T(1))/(QV*sqrt(T(1))));
    
    call(i+3) = 100*callPrice(S0, Kc(i+3), rf, 1, QV);
    delta = zeros(t+1,1);
    interest = zeros(60,1);
    cost = 0;
    shares = 0;
    for j = 2:t+1
        dlc = (log(D.Close(row1+j-1)/Kc(i+3))+(rf+QV^2/2)*T(j))/(QV*sqrt(T(j)));
        delta(j) = normcdf(dlc);
        del(j,i) = delta(j);
        shares = (delta(j)-delta(j-1))*100;
        SH(j,i) = shares;
        cost = cost+interest(j-1)+ D.Close(row1+j-1)*shares;
        C(j,i) = cost;
        interest(j) = cost*((1+rf)^(1/t)-1);
    end
    
    cost2 = 100*D.Close(row1)*(1+rf); % cost for con2
    DELTA(i+3) = delta(j);
    if D.Close(row2)<Kc(i+3)
        exe(i) = 0; 
        con1(i+3) = call(i+3);
        con2(i+3) = call(i+3)+100*D.Close(row2)-cost2;
        dyn(i+3) = call(i+3)-cost;
    else
        exe(i) = 1;
        con1(i+3) = call(i+3) + 100*(Kc(i+3)-D.Close(row2));
        con2(i+3)= call(i+3)+100*Kc(i+3)-cost2;
        dyn(i+3) = call(i+3)+100*Kc(i+3)-cost;
    end
        % convert to return rates
        dyn(i+3) = dyn(i+3)/(D.Close(row1)*100);
        con1(i+3) = con1(i+3)/(D.Close(row1)*100);
        con2(i+3) = con2(i+3)/(D.Close(row1)*100);
    
    
end

[mean(con1(4:60)),mean(con2(4:60)),mean(dyn(4:60))]
[std(con1(4:60)),std(con2(4:60)),std(dyn(4:60))]
[min(con1(4:60)),min(con2(4:60)),min(dyn(4:60))]

subplot(1,2,1); histogram(con1,'BinWidth',0.025); 
hold on
histogram(dyn);
hold off
title('Convention 1 vs. Dynamics');
legend('con1','dyn')

subplot(1,2,2); histogram(con2,'BinWidth',0.025);
hold on
histogram(dyn);
hold off

title('Convention 2 vs. Dynamics');
legend('con2','dyn')


A = table(call,Kc,DELTA,con1,con2,dyn);
table = [M,A];

%% draw the line plot
r = NaN(60,1);
for i=4:60
    r(i) = (table.Open(i)-table.Open(i-3))/table.Open(i);
end

table.r = r;
subplot(1,1,1); plot(table.Date,table.r,'--');
hold on
plot(table.Date, table.con1);
plot(table.Date, table.con2);
plot(table.Date, table.dyn,'LineWidth',1.5,'Color','k');
hold off

legend('AAPL','con1','con2','dyn');
title('Quartly return rates of AAPL and the strategies');

%% alternative test
    con1 = zeros(10000,1);
    con2 = zeros(10000,1);
    dyn = zeros(10000,1);
for i = 1:10000
    S = zeros(91,1);
    S(1) = M.Close(end);
    T = [90:-1:0]/90;
    T(end) = 0.00001;
    Kc = S(1)*1.03;
    delta = zeros(91,1);
    interest = zeros(91,1);
    call = callPrice(S(1),Kc, rf, T(1),QV);
    v = QR-1/2*QV^2-1;
    cost = 0;
    for j = 2:91
        S(j) = S(j-1)*exp(v/90+normrnd(0,1)*QV/sqrt(90));
        dlc = (log(S(j)/Kc)+(rf+QV^2/2)*T(j))/(QV*sqrt(T(j)));
        delta(j) = normcdf(dlc);
        shares = delta(j)-delta(j-1);
        cost = cost+interest(j-1)+ S(j)*shares;
        interest(j) = cost*((1+rf)^(1/90)-1);
    end
    
    if S(end)<Kc
        con1(i) = call;
        con2(i) = call+(S(end)-S(1)*(1+rf));
        dyn(i) = call-cost;
    else
        con1(i) = call+(Kc-S(end));
        con2(i) = call+(Kc-S(1)*(1+rf));
        dyn(i) = call+Kc-cost;
    end
        dyn(i) = dyn(i)/(S(1));
        con1(i) = con1(i)/(S(1));
        con2(i) = con2(i)/(S(1));
end

%% draw the histogram
subplot(1,2,1); histogram(con1); 
hold on
histogram(dyn);
hold off
title('Convention 1 vs. Dynamics');
legend('con1','dyn')

subplot(1,2,2); histogram(con2);
hold on
histogram(dyn);
hold off

title('Convention 2 vs. Dynamics');
legend('con2','dyn')

[mean(con1),mean(con2),mean(dyn)]
[std(con1),std(con2),std(dyn)]
[min(con1),min(con2),min(dyn)]


%% frequency analysis 1
Utable = zeros(60,40);
Vtable = NaN(57,40);
QR = (1+R)^3;  % quartly return
QV = sqrt(3)*V; % quartly volatility
rf = (1+0.02)^(1/4)-1; % quartly risk-free rate
for i = 1:57
    d1 = M.Date(i); % start date
    d2 = M.Date(i+3); % mature date
    row1 = find(D.Date == d1);
    row2 = find(D.Date == d2);
    t = row2-row1; % duration
    S0 = D.Close(row1);
    Kc = S0*QR;
    
    call = 100*callPrice(S0, Kc, rf, 1, QV);
    for p = 1:t/2
        delta = zeros(t+1,1);
        interest = zeros(60,1);
        shares = 100;
        delta(1) = 1;
        cost = shares*D.Close(row1);
            T0 =[t:-p:0]';  % time values for black scholes model
            if(T0(end) ~= 0)
                T = [T0;0.000001];  % correct the last entry to be non-zero
            else
                T0(end) = 0.00001;
                T = T0;
            end
            T1 =flip(T);
            T = T/t;
    for j = 2:length(T1)
        dlc = (log(D.Close(row1+T1(j))/Kc)+(rf+QV^2/2)*T(j))/(QV*sqrt(T(j)));
        delta(j) = normcdf(dlc);
        shares = (delta(j)-delta(j-1))*100;
        cost = cost+interest(j-1)+ D.Close(row1+T1(j))*shares;
        interest(j) = cost*((1+rf)^(1/t)-1);
    end
    
    if D.Close(row2)<Kc
        dyn = call-cost;
    else
         dyn= call+100*Kc-cost;
    end
        % convert to return rates
        dyn = dyn/(D.Close(row1)*100);
    Utable(i,p) = dyn;
    end
    
end

u = NaN(32,1);
for i=1:30
    u(i) = mean(Utable(:,i));
    s(i) = std(Utable(:,i));
end

subplot(1,2,1);bar(1:25, u(1:25)); title('Averge return'); xlabel('Frequency: trade every i days'); ylabel('return rate');
hold on
bar(0,0.007); legend('adjusted','original')
hold off
subplot(1,2,2);bar(1:25, s(1:25)); title('std (risk)'); xlabel('Frequency: trade every i days'); ylabel('std');
hold on
bar(0,0.0123); legend('adjusted','original')
hold off
%% Frequency analysis 2
%  alternative test
    Utable = NaN(1000,45);
for i = 1:1000
    S = zeros(91,1);
    S(1) = M.Close(end);
    for p = 1:45
        
            T0 =[90:-p:0]';  % time values for black scholes model
            if(T0(end) ~= 0)
                T = [T0;0.000001];  % correct the last entry to be non-zero
            else
                T0(end) = 0.00001;
                T = T0;
            end
            T1 =flip(T);
            T = T/90;
    Kc = S(1)*QR;
    delta = zeros(91,1);
    delta(1) = 1;
    interest = zeros(91,1);
    call = callPrice(S(1),Kc, rf, T(1),QV);
    shares = 1;
    v = QR-1/2*QV^2-1;
    cost = S(1);
    for j = 2:91
        S(j) = S(j-1)*exp(v/90+normrnd(0,1)*QV/sqrt(90));
    end
    for j = 2:length(T)
        dlc = (log(S(T1(j))/Kc)+(rf+QV^2/2)*T(j))/(QV*sqrt(T(j)));
        delta(j) = normcdf(dlc);
        shares = delta(j)-delta(j-1);
        cost = cost+interest(j-1)+ S(T1(j))*shares;
        interest(j) = cost*((1+rf)^(1/90)-1);
    end
    
    if S(end)<Kc
        dyn = call+delta(length(T))*S(end)-cost;
    else
        dyn = call+delta(length(T))*Kc-cost;
    end
        dyn = dyn/(S(1));
        Utable(i,p) = dyn;
    end
end
%%
for i=1:25
    u(i) = mean(Utable(:,i));
    s(i) = std(Utable(:,i));
end

subplot(1,2,1);bar(1:25, u(1:25)); title('Averge return'); xlabel('Frequency: trade every i days'); ylabel('return rate');
hold on
bar(0,0.0001); legend('adjusted','original')
hold off
subplot(1,2,2);bar(1:25, s(1:25)); title('std (risk)'); xlabel('Frequency: trade every i days'); ylabel('std');
hold on
bar(0,0.0201); legend('adjusted','original')
hold off
%%
function [call] = callPrice(S0, Kc, rf, T, s)

d1c = (log(S0/Kc) + (rf+s^2/2)*T)/(s*sqrt(T)); d2c = d1c-s*sqrt(T);  
call = S0*normcdf(d1c)-Kc*exp(-rf*T)*normcdf(d2c); 

end

