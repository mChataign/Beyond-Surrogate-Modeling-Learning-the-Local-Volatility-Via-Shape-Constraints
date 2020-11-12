% Import data 
% Return three vector of size n x 1 where n is the number of input data
% points
% data.T : n x 1 vector containing T-coordinates of input data set
% data.K : n x 1 vector containing K-coordinates of input data set
% data.callPrice : n x 1 vector containing observed option prices for the input coordinates

%% Premiere repartition Marc environ 50 training - 200 testing set 
%Put_testing_set = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_1/testingDataSet.csv', 'HeaderLines',1));
%Put_training_set = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_1/trainingDataSet.csv', 'HeaderLines',1));
%Interest_Dividend_Curves = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_1/dfCurve.csv','HeaderLines',1));

%% Seconde repartition Marc environ 200 training - 50 testing set -> repartition inversée 
%Put_testing_set = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_2/testingDataSet.csv', 'HeaderLines',1));
%Put_training_set = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_2/trainingDataSet.csv', 'HeaderLines',1));
%Interest_Dividend_Curves = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_2/dfCurve.csv','HeaderLines',1));

%% Troisième repartition Marc
% Put_testing_set = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_3/testingDataSet.csv', 'HeaderLines',1));
% Put_training_set = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_3/trainingDataSet.csv', 'HeaderLines',1));
% Interest_Dividend_Curves = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_3/dfCurve.csv','HeaderLines',1));

%% Quatrieme repartition Marc
Put_testing_set = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_4_50_50/testingDataSet.csv', 'HeaderLines',1));
Put_training_set = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_4_50_50/trainingDataSet.csv', 'HeaderLines',1));
Interest_Dividend_Curves = table2array(readtable('../Data/SnP500_alloc_Marc/Alloc_4_50_50/dfCurve.csv','HeaderLines',1));

%Spot_price = table2array(readtable('../Data/SnP500_alloc_Marc/7082001/underlying.csv','HeaderLines',1));

% DAX_Put_mid_training_set = table2array(readtable('../Data/data3days/7082001/trainingDataSet.csv', 'HeaderLines',1));
% DAX_Put_mid_testing_set = table2array(readtable('../Data/data3days/7082001/testingDataSet.csv', 'HeaderLines',1));
% Interest_Dividend_Curves = table2array(readtable('../Data/data3days/7082001/dfCurve.csv','HeaderLines',1));
% Spot_price = table2array(readtable('../Data/data3days/7082001/underlying.csv','HeaderLines',1));

% DAX_Put_mid_training_set = xlsread('../Data/DAX_PUT_07082001/trainingDataset_DAX_PUT_07082001.xlsx');
% DAX_Put_mid_testing_set = xlsread('../Data/DAX_PUT_07082001/testingDataset_DAX_PUT_07082001.xlsx');
% Interest_Dividend_Curves = xlsread('../Data/DAX_PUT_07082001/dfCurve.xlsx');

data.S0 = (2826.61 + 2884.22)/2; % Current price of the underlying asset.
data.r = 0;  % Annualized continuously compounded risk-free rate of return, expressed as a positive decimal number

% TRAINING DATA PUT OPTION
Maturity = Put_training_set(:,2);             % observed maturity -> T_i
Index_positive_mat = (Maturity>0);
Maturity = Maturity(Index_positive_mat);
Strike = Put_training_set(Index_positive_mat,1);               % observed strike -> K_i
ChangedStrike = Put_training_set(Index_positive_mat,9);        % modified strike -> k_i
PutPrice = Put_training_set(Index_positive_mat,3);             % observed Put price -> P(T_i, K_i)
DividendFactor = Put_training_set(Index_positive_mat,10);      % exp(\int_0^{T_i} q_t dt)
%ModifiedPutPrice = DividendFactor.*PutPrice; % \omega(T_i, k_i) = exp(\int_0^{T_i} q_t dt) P(T_i, K_i)
ModifiedPutPrice = PutPrice./DividendFactor;

% TESTING DATA PUT OPTION
Maturity_test = Put_testing_set(:,2);             % observed maturity -> T_i
Index_positive_mat = (Maturity_test>0);
Maturity_test = Maturity_test(Index_positive_mat);
Strike_test = Put_testing_set(Index_positive_mat,1);               % observed strike -> K_i
ChangedStrike_test = Put_testing_set(Index_positive_mat,9);        % modified strike -> k_i
PutPrice_test = Put_testing_set(Index_positive_mat,3);             % observed Put price -> P(T_i, K_i)
DividendFactor_test = Put_testing_set(Index_positive_mat,10);      % exp(\int_0^{T_i} q_t dt)
ModifiedPutPrice_test = PutPrice_test./DividendFactor_test; % \omega(T_i, k_i) = exp(\int_0^{T_i} q_t dt) P(T_i, K_i)


% We will perform a constrained GP regression of function \omega based on the observations : 
% (T_i, k_i) -> \omega(T_i, k_i) =  exp(\int_0^{T_i} q_t dt) P(T_i, K_i)
% given that \omega is increasing in T direction and convex in k_i

scatter3(ChangedStrike, Maturity, ModifiedPutPrice, 'MarkerEdgeColor','r', 'MarkerFaceColor','r');
hold on;
scatter3(ChangedStrike_test, Maturity_test, ModifiedPutPrice_test, 'MarkerEdgeColor','k', 'MarkerFaceColor','k');

data.T = Maturity;
data.K = ChangedStrike;
data.price = ModifiedPutPrice;
data.nb_price = length(ModifiedPutPrice);

available_strikes = ChangedStrike_test;
available_maturities = Maturity_test;
available_prices = ModifiedPutPrice_test;

%% IMPORTING Intest and Dividend Curves
int_div_time_scale = Interest_Dividend_Curves(2:end, 1);
riskFreeIntegral = Interest_Dividend_Curves(2:end, 2);
divIntegral = Interest_Dividend_Curves(2:end, 3);

riskFree_int_func = griddedInterpolant(int_div_time_scale,riskFreeIntegral); % create interpolant function object riskFree_int
div_int_func = griddedInterpolant(int_div_time_scale,divIntegral); % create interpolant function object div_int_func

% plot(int_div_time_scale, riskFreeIntegral);
% hold on;
% t=0:0.001:1;
% y = riskFree_int_func(t);
% plot(t,y);

% plot(int_div_time_scale, divIntegral);
% hold on;
% t=0:0.001:1;
% y = div_int_func(t);
% plot(t,y);


