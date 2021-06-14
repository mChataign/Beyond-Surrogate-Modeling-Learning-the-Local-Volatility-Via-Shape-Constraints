% Import data 
% Return three vector of size n x 1 where n is the number of input data
% points
% data.T : n x 1 vector containing T-coordinates of input data set
% data.K : n x 1 vector containing K-coordinates of input data set
% data.callPrice : n x 1 vector containing observed option prices for the input coordinates

Put_testing_set = table2array(readtable('testingDataSet.csv', 'HeaderLines',1));
Put_training_set = table2array(readtable('trainingDataSet.csv', 'HeaderLines',1));
Interest_Dividend_Curves = table2array(readtable('dfCurve.csv','HeaderLines',1));



%Spot_price = table2array(readtable('../Data/SnP500_alloc_Marc/7082001/underlying.csv','HeaderLines',1));

% DAX_Put_mid_training_set = table2array(readtable('../Data/data3days/7082001/trainingDataSet.csv', 'HeaderLines',1));
% DAX_Put_mid_testing_set = table2array(readtable('../Data/data3days/7082001/testingDataSet.csv', 'HeaderLines',1));
% Interest_Dividend_Curves = table2array(readtable('../Data/data3days/7082001/dfCurve.csv','HeaderLines',1));
% Spot_price = table2array(readtable('../Data/data3days/7082001/underlying.csv','HeaderLines',1));

% DAX_Put_mid_training_set = xlsread('../Data/DAX_PUT_07082001/trainingDataset_DAX_PUT_07082001.xlsx');
% DAX_Put_mid_testing_set = xlsread('../Data/DAX_PUT_07082001/testingDataset_DAX_PUT_07082001.xlsx');
% Interest_Dividend_Curves = xlsread('../Data/DAX_PUT_07082001/dfCurve.xlsx');

%data.S0 = (2826.61 + 2884.22)/2; % Current price of the underlying asset.
data.S0 = 2859.53;
data.r = 0;  % Annualized continuously compounded risk-free rate of return, expressed as a positive decimal number

% TRAINING DATA PUT OPTION
Maturity = Put_training_set(:,2);             % observed maturity -> T_i
Index_positive_mat = (Maturity>1);
%Index_positive_mat = (Maturity>0.05);
%Index_positive_mat = (Maturity>0.056);
%Index_positive_mat = (Maturity>0.08);
Maturity = Maturity(Index_positive_mat);
Strike = Put_training_set(Index_positive_mat,1);               % observed strike -> K_i
ChangedStrike = Put_training_set(Index_positive_mat,9);        % modified strike -> k_i
PutPrice_mid = Put_training_set(Index_positive_mat,3);             % observed Put price -> P(T_i, K_i)
PutPrice_bid = Put_training_set(Index_positive_mat,18);             % observed Put price -> P(T_i, K_i)
PutPrice_ask = Put_training_set(Index_positive_mat,19);             % observed Put price -> P(T_i, K_i)
DividendFactor = Put_training_set(Index_positive_mat,10);      % exp(\int_0^{T_i} q_t dt)
%ModifiedPutPrice = DividendFactor.*PutPrice; % \omega(T_i, k_i) = exp(\int_0^{T_i} q_t dt) P(T_i, K_i)
ModifiedPutPrice_mid = PutPrice_mid./DividendFactor;
ModifiedPutPrice_bid = PutPrice_bid./DividendFactor;
ModifiedPutPrice_ask = PutPrice_ask./DividendFactor;


% ---------- Update to incorporate replicates -----------
ChangedStrike_bid_ask = [ ChangedStrike; ChangedStrike];
Maturity_bid_ask = [ Maturity; Maturity];
ModifiedPutPrice_bid_ask = [ModifiedPutPrice_bid; ModifiedPutPrice_ask];

% % ModifiedPutPrice_bid_ask = [ModifiedPutPrice_bid_ask; ModifiedPutPrice_bid(Index_low_price) ; ModifiedPutPrice_ask(Index_low_price)];

% TESTING DATA PUT OPTION
Maturity_test = Put_testing_set(:,2);             % observed maturity -> T_i
Index_positive_mat = (Maturity_test>1);
%Index_positive_mat = (Maturity_test>0.05);
%Index_positive_mat = (Maturity_test>0.056);
%Index_positive_mat = (Maturity_test>0.08);

Maturity_test = Maturity_test(Index_positive_mat);
Strike_test = Put_testing_set(Index_positive_mat,1);               % observed strike -> K_i
ChangedStrike_test = Put_testing_set(Index_positive_mat,9);        % modified strike -> k_i
PutPrice_test_mid = Put_testing_set(Index_positive_mat,3);             % observed Put price -> P(T_i, K_i)
PutPrice_test_bid = Put_testing_set(Index_positive_mat,18);             % observed Put price -> P(T_i, K_i)
PutPrice_test_ask = Put_testing_set(Index_positive_mat,19);             % observed Put price -> P(T_i, K_i)
DividendFactor_test = Put_testing_set(Index_positive_mat,10);      % exp(\int_0^{T_i} q_t dt)
ModifiedPutPrice_test_mid = PutPrice_test_mid./DividendFactor_test; % \omega(T_i, k_i) = exp(\int_0^{T_i} q_t dt) P(T_i, K_i)
ModifiedPutPrice_test_bid = PutPrice_test_bid./DividendFactor_test; % \omega(T_i, k_i) = exp(\int_0^{T_i} q_t dt) P(T_i, K_i)
ModifiedPutPrice_test_ask = PutPrice_test_ask./DividendFactor_test; % \omega(T_i, k_i) = exp(\int_0^{T_i} q_t dt) P(T_i, K_i)

% ---------- Update to incorporate replicates -----------
ChangedStrike_test_bid_ask = [ ChangedStrike_test; ChangedStrike_test];
Maturity_test_bid_ask = [ Maturity_test; Maturity_test];
ModifiedPutPrice_test_bid_ask = [ModifiedPutPrice_test_bid; ModifiedPutPrice_test_ask];

% We will perform a constrained GP regression of function \omega based on the observations : 
% (T_i, k_i) -> \omega(T_i, k_i) =  exp(\int_0^{T_i} q_t dt) P(T_i, K_i)
% given that \omega is increasing in T direction and convex in k_i


figure;
scatter3(ChangedStrike, Maturity, ModifiedPutPrice_ask-ModifiedPutPrice_bid, 'MarkerEdgeColor','r', 'MarkerFaceColor','r');
hold on;
scatter3(ChangedStrike_test, Maturity_test, ModifiedPutPrice_test_ask - ModifiedPutPrice_test_bid, 'MarkerEdgeColor','k', 'MarkerFaceColor','k');


figure;
scatter3(ChangedStrike, Maturity, (ModifiedPutPrice_ask-ModifiedPutPrice_bid)./ModifiedPutPrice_ask, 'MarkerEdgeColor','r', 'MarkerFaceColor','r');
hold on;
scatter3(ChangedStrike_test, Maturity_test, (ModifiedPutPrice_test_ask - ModifiedPutPrice_test_bid)./ModifiedPutPrice_test_ask, 'MarkerEdgeColor','k', 'MarkerFaceColor','k');


figure;
scatter3(ChangedStrike_bid_ask, Maturity_bid_ask, ModifiedPutPrice_bid_ask, 'MarkerEdgeColor','r', 'MarkerFaceColor','r');
hold on;
scatter3(ChangedStrike_test_bid_ask, Maturity_test_bid_ask, ModifiedPutPrice_test_bid_ask, 'MarkerEdgeColor','k', 'MarkerFaceColor','k');


data.T = Maturity_bid_ask;
data.K = ChangedStrike_bid_ask;
data.price = ModifiedPutPrice_bid_ask;
data.nb_price = length(ModifiedPutPrice_bid_ask);

available_strikes = ChangedStrike_test_bid_ask;
available_maturities = Maturity_test_bid_ask;
available_prices = ModifiedPutPrice_test_bid_ask;

%% IMPORTING Intest and Dividend Curves
int_div_time_scale = Interest_Dividend_Curves(2:end, 1);
riskFreeIntegral = Interest_Dividend_Curves(2:end, 2);
divIntegral = Interest_Dividend_Curves(2:end, 3);

riskFree_int_func = griddedInterpolant(int_div_time_scale,riskFreeIntegral); % create interpolant function object riskFree_int
div_int_func = griddedInterpolant(int_div_time_scale,divIntegral); % create interpolant function object div_int_func

