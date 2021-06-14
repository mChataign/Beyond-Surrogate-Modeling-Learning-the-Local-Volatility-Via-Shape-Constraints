%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 		
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    Kriging_vol_surface_scattered_data.m
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  In this program, we generate a set of sampled arbitrage-free volatility surfaces 
%  compatible with some input noisy option prices for different strike and time-to-maturities
%  A finite-dimensional approximation of the original GP prior is used to handle monotonicity and
%  convexity constraints on the entire domain. The prior process is then
%  caracterized by a Gaussian vector \xi with exactly the same number of elements that knots in 
%  the basis function grid. Let X be a set of points in the domain, the approximated Gaussian process prior is 
%  Y^N(X) =\Phi(X) \xi, where \Phi(X) is a matrix of basis functions.
%  The kernel hyper-parameters and the noise variance are estimated/learned on the input data using maximum likelihood.
%  We consider two versions of the likelihood : unconstrained vs constrained
%  We use Hamilton Monte Carlo method to sample paths of the truncated Gaussian process and to quantify uncertainty.
%  The program proceed as follows : 
% 1. Import input observed data - non necessarly gridded data 
% 2. Construct the grid of basis functions - non necessarly a grid with
% constant step (for each direction)
%    a. Define the input domain on which we want to study the phenomenon
%    b. Construct the grid on that domain
%    c. Normalized the input domain on [0,1]^d
% 3. Construct the constraint matrices (matrices that caracterized
% inequality constraints on the finite-dimensional approximation of the
% Gaussian process)
% 4. Construct the equality constraint matrix A_eq =\phi(X) and B_eq
%    A_eq is a n x N matrix, B_eq is a n x 1 vector
% 5. Maximize the log-likelohood using multistart or global optim starting
% from a range of admissible hyper-parameter
%   a. Using unconstrained log-likelihood 
%   b. Using full constrained log-likelihood
% 6. Convert optimal scale hyper-parameters (the theta's) back to the original input grid 
% 7. Construct the MAP (maximum a posteriori) for Y^N and the noise
% structure \epsilon -> Give most probable a priori surface and noise srtucture
% 8. Sample Nbsimul paths of the truncated Gaussian process using exact HMC
% method
% 9. Compute [0.05 - 0.95] quantile surfaces 
%
%
%  Created on 2019-05-17, at 9:40, by Areski Cousin
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clc
%clear all
%close all

%% 0. Program global parameter 

unconstr_like = true; %true if unconstrained likelihood is maximized, false for constrained likelihood
%unconstr_like = false; %true if unconstrained likelihood is maximized, false for constrained likelihood
Nb_simu_HCM = 1;


%%  1. Import input observed data
disp('Importing data...');
% disp(['Generating random ' num2str(matdim) ' x ' num2str(matdim) ...
%     ' symmetric matrix....']);

Import_data_scattered_SnP500_18_05_2019
% Gives :
% data.T
% data.K
% data.price
% in a future version with scatter input observation : 
% gives data.T_min, data.T_max, data.K_min, data.K_max and scatter data in
% the form of three vector data.T, data.K, data.price
% [data.T data.K] contains the coordinates of input locations
% data.price contains obseved output values at the input locations

%% 2. Construct the grid of basis functions
disp('Constructing the grid of basis function...');
% a. Define the input space domain
%x_nodes.nb = 30; % Specify the number of x_nodes
%x_nodes.nb = 70;
%x_nodes.nb = 25; % test OK 2 - plus rapide
%x_nodes.nb = 60; % test 1
x_nodes.nb = 30; % test OK
x_nodes.min = min([available_strikes; data.K]);
%x_nodes.min = min(data.K)-20; %Important -> allows to rescaled back the input domain -> min(data.K)-20 for scatter data
x_nodes.max = max([available_strikes; data.K]);

x_nodes.vect = linspace(x_nodes.min, x_nodes.max, x_nodes.nb); %raw grid (not normalized on [0,1])

% Raffiner la grille autour de la monnaie -> les contraintes d'AOA sont à
% adapter pour cela
% x_grid_heart = linspace(2000, 3100, 25);
% x_nodes.vect = [x_nodes.vect x_grid_heart];
% x_nodes.min = min(x_nodes.vect);
% x_nodes.max = max(x_nodes.vect);
% x_nodes.nb = length(x_nodes.vect);

%x_nodes.max = max(data.K)+10; % -> max(data.K)+10 for scatter data
x_nodes.length = x_nodes.max-x_nodes.min; %Important -> allows to rescaled back the input domain


x_nodes.delta = x_nodes.vect(2:end)-x_nodes.vect(1:(end-1)); % possiblité d'intégrer une grille non-homogène


t_nodes.nb = 30; % test OK
%t_nodes.nb = 40; % test 1
%t_nodes.nb = 40; % test 2
%t_nodes.min = min(data.T)-0.01; %Important -> allows to rescaled back the input domain
t_nodes.min = min([available_maturities; data.T]);
%t_nodes.min = 0;
%t_nodes.min = min(data.T); %Important -> allows to rescaled back the input domain
t_nodes.max = max([available_maturities; data.T]);
%t_nodes.max = max(available_maturities)+2;
%t_nodes.max = max(data.T)+0.01;
%t_nodes.max = t_nodes.max+1;
t_nodes.length = t_nodes.max-t_nodes.min; %Important -> allows to rescaled back the input domain
t_nodes.vect = linspace(t_nodes.min, t_nodes.max, t_nodes.nb); %raw grid (not normalized on [0,1])
t_nodes.delta = t_nodes.vect(2:end)-t_nodes.vect(1:(end-1));

data.nb_nodes = x_nodes.nb*t_nodes.nb;

% b. Renormalization of the input space on [0,1]^d
x_nodes_scaled.nb = x_nodes.nb;
x_nodes_scaled.vect = (x_nodes.vect -  x_nodes.min)./x_nodes.length;
x_nodes_scaled.delta = x_nodes_scaled.vect(2:end)-x_nodes_scaled.vect(1:(end-1)); % possiblité d'intégrer une grille non-homogène

t_nodes_scaled.nb = t_nodes.nb;
t_nodes_scaled.vect = (t_nodes.vect - t_nodes.min)./t_nodes.length;
t_nodes_scaled.delta = t_nodes_scaled.vect(2:end)-t_nodes_scaled.vect(1:(end-1));

data.x_nodes_scaled = x_nodes_scaled;
data.t_nodes_scaled = t_nodes_scaled;

% c. Renormalization of the input observed points
data.K_scaled = (data.K -  x_nodes.min)./x_nodes.length;
data.T_scaled = (data.T - t_nodes.min)./t_nodes.length;


%% 3. Construct the inequality constraint matrices
disp('Constructing the inequality constraint matrices...');
tic

disp('Constructing the convexity constraint matrices...');
A = [];
B = [];


% condition 1) convexity in strike
for i=2:x_nodes.nb-1 
    for j=0:t_nodes.nb-1 
       current_row_A = zeros(1,data.nb_nodes);
       current_row_A(1, i*t_nodes.nb+j+1) = 1;
       current_row_A(1, (i-1)*t_nodes.nb+j+1) = -2;
       current_row_A(1, (i-2)*t_nodes.nb+j+1) = 1;
       A = [A; current_row_A];  
       B = [B; 0];         
    end
end
   
disp('Constructing the increasing in strike constraint matrices...');
% condition 1bis a) for Put Options : price increasing in strike


% Only needed to be tested at K_min
for j=0:t_nodes.nb-1 
   current_row_A = zeros(1,data.nb_nodes);
   current_row_A(1, t_nodes.nb+j+1) = 1; %i=1
   current_row_A(1, j+1) = -1; %i=0
   A = [A; current_row_A];  
   B = [B; 0];         
end

% for i=1:x_nodes.nb-1 
%     for j=0:t_nodes.nb-1 
%        current_row_A = zeros(1,data.nb_nodes);
%        current_row_A(1, i*t_nodes.nb+j+1) = 1;
%        current_row_A(1, (i-1)*t_nodes.nb+j+1) = -1;
%        A = [A; current_row_A];  
%        B = [B; 0];         
%     end
% end

% condition 1bis b) for Call Options : price decreasing in strike
% for i=1:x_nodes.nb-1 
%     for j=0:t_nodes.nb-1 
%        current_row_A = zeros(1,data.nb_nodes);
%        current_row_A(1, i*t_nodes.nb+j+1) = -1;
%        current_row_A(1, (i-1)*t_nodes.nb+j+1) = 1;
%        A = [A; current_row_A];  
%        B = [B; 0];         
%     end
% end

disp('Constructing the increasing in maturity constraint matrices...');
% condition 2) increasingness in maturity

for i=0:x_nodes.nb-1 
    for j=1:t_nodes.nb-1 
       current_row_A = zeros(1,data.nb_nodes);
       current_row_A(1, i*t_nodes.nb+j+1) = 1;
       current_row_A(1, i*t_nodes.nb+j) = -1;
       A = [A; current_row_A];  
       B = [B; 0];         
    end
end

% condition 3) surface values should be positive.
% Given that the surface is increasing in maturity direction, 
% this condition can be checked only for the points 
% in the basis function grid with t = t_nodes.min
disp('Constructing the positivity constraint matrices...');
for i=0:x_nodes.nb-1 
   current_row_A = zeros(1,data.nb_nodes);
   current_row_A(1, i*t_nodes.nb+1) = 1;
   A = [A; current_row_A];  
   B = [B; 0];
end

data.A = A;
data.B = B;
data.Nbconstr = length(B);
toc %very slow - speed can be improved considering tensor product construction - see the hand notes


%% 4. Construct the equality constraint matrix
disp('Constructing the equality constraint matrices...');

% Scattered data version :
Phi = Basis_func_scattered_data(data.K_scaled, data.T_scaled, data.x_nodes_scaled, data.t_nodes_scaled);
% Old version
% [Phi1, Phi2] = Basis_func_decomp(data.K_scaled, data.T_scaled, data.x_nodes_scaled, data.t_nodes_scaled); %does not depend on hyperparameter
% data.Phi1 = Phi1;
% data.Phi2 = Phi2;
% Phi = kron(data.Phi1, data.Phi2);
Price_vector = data.price;


% % % % Put such that P(S_0, 0) = (K-S_0)^+
% for i=0:x_nodes.nb-1 
%     current_row_Aeq = zeros(1,data.nb_nodes); 
%     current_row_Aeq(1,i*t_nodes.nb+1) = 1;
%     current_row_Beq = max(0,x_nodes.vect(i+1) - data.S0); 
%     Phi = [Phi; current_row_Aeq]; 
%     Price_vector = [Price_vector; current_row_Beq]; 
%     data.nb_price = data.nb_price +1;
% end

Aeq = Phi;
Beq = Price_vector;
data.Phi = Phi;
%data.price = Price_vector;

% % % % % Call such that C(S_0, 0) = (S_0-K)^+
% for i=0:x_nodes.nb-1 
%     current_row_Aeq = zeros(1,data.nb_nodes); 
%     current_row_Aeq(1,i*t_nodes.nb+1) = 1;
%     current_row_Beq = max(0, data.S0 - x_nodes.vect(i+1)); 
%     Aeq = [Aeq; current_row_Aeq]; 
%     Beq = [Beq; current_row_Beq]; 
% end

% % % %condition 3) 
% % % for i=1:t_nodes.nb    
% % %     current_row_Aeq = zeros(1,data.nb_nodes); 
% % %     current_row_Aeq(1,t_nodes.nb+i) = 1;
% % %     current_row_Beq = data.S0 - x_nodes.delta(1)*exp(-data.r*t_nodes.vect(i)); 
% % %     Aeq = [Aeq; current_row_Aeq];     
% % %     Beq = [Beq; current_row_Beq];      
% % % end

data.nb_eq_constr = length(Beq);    
if (data.nb_eq_constr>=data.nb_nodes)
    disp('Problem : the number of nodes should be greater than the number of constraints');
end
   

%% 5. Maximize the log-likelohood 
disp('Maximizing the log-likelohood...');
% Computation of max log-likelihood version (with one single noise parameter)

% Initial parameters
sigma = 5;
theta_1 = 0.1;
theta_2 = 0.3;
noise1 = 1;
param_init = [sigma, theta_1, theta_2, noise1];
%Log_like = log_likelihood_unconstr_single_noise_param(param_init, data);

% Objective function handle

if unconstr_like
    Log_like_handle = @(x)(-log_likelihood_unconstr_single_noise_param_scatter(x, Price_vector, data)); %we take the opposite since we want to maximize
else
    Log_like_handle = @(x)(-log_likelihood_constr_single_noise_param_scatter(x, Price_vector, data)); %we take the opposite since we want to maximize 
end

% Definition of the exploratory domain for hyper-parameter
sigma_l = 0;
sigma_u = 1000;
theta_1_l = 0;
theta_1_u = 2;
theta_2_l = 0;
theta_2_u = 2;
noise1_l = 0;
noise1_u = 10;

lower_bound = [sigma_l, theta_1_l, theta_2_l, noise1_l];
upper_bound = [sigma_u, theta_1_u, theta_2_u, noise1_u];

tic

% MultiStart
%opts = optimoptions(@fmincon,'Algorithm','sqp');
problem = createOptimProblem('fmincon','objective',Log_like_handle,...
            'x0',param_init,'lb',lower_bound,'ub',upper_bound);%,'options',opts);
ms = MultiStart('Display', 'iter');
%ms = MultiStart('Display', 'off');
%NB_start = 10;
NB_start = 1;
[x,f] = run(ms,problem,NB_start);

toc

data.sigma_opt = x(1);
data.theta_1_opt = x(2);
data.theta_2_opt = x(3);
data.noise1_opt = x(4);

sigma_opt = x(1)
theta_1_opt = x(2)
theta_2_opt = x(3)
noise1_opt = x(4)

% Global Search Algo
% gs = GlobalSearch('Display', 'iter');
% Log_like_handle = @(x)(-log_likelihood_unconstr_2(x, data)); %we take the opposite since we want to maximize
% %Log_like_handle = @(x)(-log_likelihood_part_2(x, data)); %we take the opposite since we want to maximize
%
% problem = createOptimProblem('fmincon','objective',Log_like_handle,...
%     'x0',param_init,'lb',lower_bound,'ub',upper_bound);
% x = run(gs,problem);

%% 6. Convert optimal scaled hyper-parameters (the theta's) back to the original input grid 
% For inforamtion purpose only
disp('Convert optimal scaled hyper-parameters back to the original input grid...');
Sigma_opt = x(1)
Theta_1_opt_rescaled_back = data.theta_1_opt*x_nodes.length
Theta_2_opt_rescaled_back = data.theta_2_opt*t_nodes.length
Noise_opt = x(4)

data.theta_1_opt_rescaled_back = Theta_1_opt_rescaled_back;
data.theta_2_opt_rescaled_back = Theta_2_opt_rescaled_back;

%% 7. Construct the MAP (maximum a posteriori) for Y^N and the noise
disp('Constructing the MAP (maximum a posteriori) for Y^N and the noise...');

% Construction of covariance matrix Gamma of $\xi$
[Gamma1, Gamma2] = Gamma_decomp(data.sigma_opt, data.theta_1_opt, data.theta_2_opt, data);
Gamma = kron(Gamma1, Gamma2);
Gamma = (Gamma + Gamma')/2;
%nugget = 0;
%nugget = 1e-8;
nugget = 1e-6;
Gamma = Gamma + nugget*eye(data.nb_nodes);

% Choleski inverstion of Gamma
invGamma = invChol_mex(Gamma);

% Definition of the noise matrix 
% should be updated if there is more observation than input location points
% Implicitly assume that there is a unique observation per input location
Sigma_noise =  data.noise1_opt*eye(data.nb_price);
% Inversion of the diagonal noise matrix
inv_Sigma_noise = diag(diag(1./Sigma_noise));

Nb_observ = data.nb_price;

Quad_matrix = blkdiag(invGamma, inv_Sigma_noise);
Aeq = [Aeq eye(Nb_observ)];
nb_row_A = size(data.A,1);
A_new = [data.A zeros(nb_row_A,Nb_observ)];
B_new = data.B;
f = zeros(data.nb_nodes+Nb_observ,1);

%%
tic
opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','TolCon', 1e-12, 'Display','iter');
%opts = optimoptions('quadprog',...
%    'Algorithm','interior-point-convex','TolCon', 1e-14, 'Display','iter');
% opts = optimoptions('quadprog',...
%     'Algorithm','interior-point-convex','TolCon', 1e-15, 'Display','iter');
% opts = optimoptions('quadprog',...
%     'Algorithm','interior-point-convex','TolCon', 1e-12, 'Display','iter');
% opts = optimoptions('quadprog',...
%      'Algorithm','trust-region-reflective','Display','iter'); %not possible with the specified constraints
% To improve quadprog, start with a linear interpolation of the observation
% points and use an algorithm that satisfies the constrains at each
% iteration steps.
%[Out_put, fval] = quadprog(Quad_matrix,f,-A_new,B_new,Aeq,Beq, [], [], [], opts);   

Tol_AOA_constr = 1e-5;
%Tol_AOA_constr = 1e-10;
[Out_put, fval] = quadprog(Quad_matrix,f,-A_new,B_new-Tol_AOA_constr,Aeq,Beq, [], [], [], opts);   

toc
Xi_mode = Out_put(1:data.nb_nodes);
Most_probable_noise_values = Out_put((data.nb_nodes+1):end);
fval

%% Check that inequality constraints are satistfied by the mode
Constraint_cond_min = min(A*Xi_mode)

% Former version of the MAP : 
% [Xi_mode, fval] = quadprog(invGamma,f,-A,B,Aeq,Beq, [], [], [], opts);    
% Xi_mode
% fval

%% Compute RMSE on training set
RMSE_bid_ask = sqrt(mean((data.Phi*Xi_mode - Beq).^2))
RMSE_rel_bid_ask = mean(abs((data.Phi*Xi_mode - Beq)./Beq))
% RMSE on mid price only
K_scaled_mid_only = (ChangedStrike -  x_nodes.min)./x_nodes.length;
T_scaled_mid_only = (Maturity - t_nodes.min)./t_nodes.length;
Phi_mid_point = Basis_func_scattered_data(K_scaled_mid_only, T_scaled_mid_only, data.x_nodes_scaled, data.t_nodes_scaled);
RMSE_mid = sqrt(mean((Phi_mid_point*Xi_mode - ModifiedPutPrice_mid).^2))
RMSE_rel_mid = mean(abs((Phi_mid_point*Xi_mode - ModifiedPutPrice_mid)./ModifiedPutPrice_mid))


%% Compute RMSE on testing set

% Renormalization of the input observed points
K_test_scaled = (available_strikes -  x_nodes.min)./x_nodes.length;
T_test_scaled = (available_maturities - t_nodes.min)./t_nodes.length;
Phi_test_point = Basis_func_scattered_data(K_test_scaled, T_test_scaled, data.x_nodes_scaled, data.t_nodes_scaled);
Phi_test_point_2 = Basis_func_scattered_data(available_strikes, available_maturities, x_nodes, t_nodes);
RMSE_testset = sqrt(mean((Phi_test_point*Xi_mode - available_prices).^2))
%[Phi_test_point*Xi_mode available_prices]
%RMSE_testset = sqrt(mean((Phi_test_point_2*Xi_mode - available_prices).^2)); %gives same result
RMSE_rel_testset = mean(abs((Phi_test_point*Xi_mode - available_prices)./available_prices))
K_scaled_mid_only_testset = (ChangedStrike_test -  x_nodes.min)./x_nodes.length;
T_scaled_mid_only_testset = (Maturity_test - t_nodes.min)./t_nodes.length;
Phi_mid_point_testset = Basis_func_scattered_data(K_scaled_mid_only_testset, T_scaled_mid_only_testset, data.x_nodes_scaled, data.t_nodes_scaled);
RMSE_mid_testset = sqrt(mean((Phi_mid_point_testset*Xi_mode - ModifiedPutPrice_test_mid).^2))
RMSE_rel_mid_testset = mean(abs((Phi_mid_point_testset*Xi_mode - ModifiedPutPrice_test_mid)./ModifiedPutPrice_test_mid))

%% plot MAP function on the initial domain -> need to scale back
x = linspace(x_nodes.min, x_nodes.max, 50);
t = linspace(t_nodes.min, t_nodes.max, 50);
[xx, tt] = meshgrid(x,t);
n_x = length(x);
n_t = length(t);
% Compute the finite-dimensional Gaussian process for points in x and t
[Phi_x, Phi_t] = Basis_func_decomp(x, t, x_nodes, t_nodes);
Phi_xt = kron(Phi_x,Phi_t);
Y = Phi_xt*Xi_mode;

figure; % strike first
hold on;
surf(xx, tt, reshape(Y, n_t, n_x));
% plot observed price
scatter3(available_strikes,available_maturities, available_prices, 'MarkerEdgeColor','k', 'MarkerFaceColor','k');
hold on;
scatter3(data.K, data.T, data.price, 'MarkerEdgeColor','r', 'MarkerFaceColor','r');
axis tight; grid on;
title('Put Price Surface','Fontsize',16,'FontWeight','Bold','interpreter','latex');
xlabel('Modified Strike','Fontsize',16,'FontWeight','Bold','interpreter','latex');
ylabel('Maturity','Fontsize',16,'FontWeight','Bold','interpreter','latex');
zlabel('Modified Put Price','Fontsize',16,'FontWeight','Bold','interpreter','latex');
set(gca,'Fontsize',16,'LineWidth',1);


%% Plot most probable noise values
figure;
scatter3(data.K, data.T, Most_probable_noise_values, 'MarkerEdgeColor','r', 'MarkerFaceColor','r');
%scatter3(K_row, T_row, price_row+Most_probable_noise_values', 'MarkerEdgeColor','r', 'MarkerFaceColor','r');
axis tight; grid on;
title('Most Probable Noise Values','Fontsize',18,'FontWeight','Bold');
xlabel('Modified Strike','Fontsize',18,'FontWeight','Bold');
ylabel('Maturity','Fontsize',18,'FontWeight','Bold');
zlabel('Noise Value','Fontsize',18,'FontWeight','Bold');
set(gca,'Fontsize',18,'LineWidth',1);


%% Plot MAP and at each maturities 

Observed_maturities = unique(intersect(data.T, available_maturities));
Nb_obs_mat = size(Observed_maturities, 1);

for i=1:Nb_obs_mat
% Compute the finite-dimensional Gaussian process for points in x and t

    figure;

    hold on;
    index_available_maturities_at_T = (available_maturities == Observed_maturities(i));
    scatter(available_strikes(index_available_maturities_at_T), available_prices(index_available_maturities_at_T), 'MarkerEdgeColor','k', 'MarkerFaceColor','k');
    
    index_traning_maturities_at_T = (data.T == Observed_maturities(i));
    scatter(data.K(index_traning_maturities_at_T), data.price(index_traning_maturities_at_T), 'MarkerEdgeColor','r', 'MarkerFaceColor','r')

    %available_strikes(index_available_maturities_at_T)
    %data.K(index_traning_maturities_at_T)
    
    x_min = min(available_strikes(index_available_maturities_at_T));
    x_min = min(x_min, min(data.K(index_traning_maturities_at_T)));
    x_max = max(available_strikes(index_available_maturities_at_T));
    x_max = max(x_max, max(data.K(index_traning_maturities_at_T)));
    
    x = linspace(x_min, x_max, 50);
    %n_x = length(x);
    
    [Phi_x, Phi_t] = Basis_func_decomp(x, Observed_maturities(i), x_nodes, t_nodes);
    Phi_xt = kron(Phi_x,Phi_t);
    Y = Phi_xt*Xi_mode;
    
    plot(x, Y);
    title_fig = strcat('Put Price at maturity T = ', ' ', num2str(Observed_maturities(i)));
    title(title_fig,'Fontsize',16,'FontWeight','Bold','interpreter','latex');
    
    
    axis tight; grid on;
    
    %pause;
end

%% Plot Implied Vol from MAP and at each maturities 

Observed_maturities = unique(intersect(data.T, available_maturities)); %maturite unique observe a la fois en training et testing
Nb_obs_mat = size(Observed_maturities, 1);

for i=1:Nb_obs_mat
% Compute the finite-dimensional Gaussian process for points in x and t

    figure;
    hold on; 
    
    % scatter plot of IV at testing points (bid and ask IV of testing set)
    index_available_maturities_at_T = (available_maturities == Observed_maturities(i)); % indice du testing set (bid-ask) avec la maturité i
    T_vol = available_maturities(index_available_maturities_at_T); %replication de la même maturité i pour tout les indices du test set avec mat i
    r_int_vect = riskFree_int_func(T_vol);
    div_int_vect = div_int_func(T_vol);
    Put_price_vect = exp(-div_int_vect).*available_prices(index_available_maturities_at_T);
    [T, K_test, IV] = Vol_from_putPrice_with_div(data.S0, r_int_vect, div_int_vect, available_strikes(index_available_maturities_at_T), T_vol, Put_price_vect);
    scatter(K_test, IV, 'MarkerEdgeColor','k', 'MarkerFaceColor','k');
    
    % scatter plot of IV at training points (bid and ask IV of training set)
    index_traning_maturities_at_T = (data.T == Observed_maturities(i));
    T_vol = data.T(index_traning_maturities_at_T); %replication de la même maturité i pour tout les indices du test set avec mat i
    r_int_vect = riskFree_int_func(T_vol);
    div_int_vect = div_int_func(T_vol);
    Put_price_vect = exp(-div_int_vect).*data.price(index_traning_maturities_at_T);
    [T, K_train, IV] = Vol_from_putPrice_with_div(data.S0, r_int_vect, div_int_vect, data.K(index_traning_maturities_at_T), T_vol, Put_price_vect);
    scatter(K_train, IV, 'MarkerEdgeColor','r', 'MarkerFaceColor','r');
        
    x_min = min(min(K_test), min(K_train));
    x_max = max(max(K_test), max(K_train));
    
    nb_x = 50;
    x = linspace(x_min, x_max, nb_x);
    %n_x = length(x);
    
    [Phi_x, Phi_t] = Basis_func_decomp(x, Observed_maturities(i), x_nodes, t_nodes);
    Phi_xt = kron(Phi_x,Phi_t);
    Y = Phi_xt*Xi_mode;
    
    T_vol = repmat(Observed_maturities(i), 1, nb_x);
    r_int_vect = riskFree_int_func(T_vol);
    div_int_vect = div_int_func(T_vol);
    Put_price_vect = exp(-div_int_vect).*Y;
    [T, K_GP, IV] = Vol_from_putPrice_with_div(data.S0, r_int_vect, div_int_vect, x, T_vol, Put_price_vect);
    plot(K_GP, IV);
    
    title_fig = strcat('Implied Vol at maturity T = ', ' ', num2str(Observed_maturities(i)));
    title(title_fig,'Fontsize',16,'FontWeight','Bold','interpreter','latex');
    
    axis tight; grid on;
    
    %pause;
end



%% Plot Dupire vol surface

%n_x = 70;

n_x = 20;
%n_x = floor(x_nodes.nb./2)-40; %FSTE 02/12/1999
%n_x = floor(x_nodes.nb./2)+10; %09/08/2001
%n_x = floor(x_nodes.nb./2)+5; %08/08/2001
%n_x = floor(x_nodes.nb./2)+8; %07/08/2001

n_t = 20;

% x_k = linspace(min(available_strikes), max(available_strikes), n_x); %grille des strikes modifiés de taille n_x sur l'enveloppe convexe testset
% x = linspace(min(Strike_test), max(Strike_test), n_x); %grille des strikes (non-modifiés) de taille n_x sur l'enveloppe convexe testset

Strike_min = min([available_strikes; ChangedStrike]);
Strike_max = max([available_strikes; ChangedStrike]);
x_k = linspace(Strike_min, Strike_max, n_x); %grille des strikes modifiés de taille n_x sur l'enveloppe convexe testset

%t = linspace(min(available_maturities), max(available_maturities), n_t); %grille des maturités de taille n_t sur l'enveloppe convexe testset

Maturity_min = min([available_maturities; Maturity]);
%Maturity_min = min(available_maturities);
%Maturity_min = 0.15;
%Maturity_min = 0.5;

Maturity_max = max([available_maturities; Maturity]);
%Maturity_max = max(available_maturities);
%Maturity_max = 2;


t = linspace(Maturity_min, Maturity_max, n_t); %grille des maturités de taille n_t sur l'enveloppe convexe testset

%t = linspace(min(available_maturities), max(available_maturities), n_t); %grille des maturités de taille n_t sur l'enveloppe convexe testset
% n_x = 60;
% n_t = 20;
% x = linspace(x_nodes.min, x_nodes.max, n_x);
% t = linspace(t_nodes.min+0.2, t_nodes.max, n_t);

step_size_x = x_k(2)-x_k(1); %pas de la grille en K modifié
step_size_t = t(2)-t(1); %pas de la grille en T

% Compute the finite-dimensional Gaussian process for points in x and t %
[Phi_x, Phi_t] = Basis_func_decomp(x_k, t, x_nodes, t_nodes);
Phi_xt = kron(Phi_x,Phi_t);
Y = Phi_xt*Xi_mode; %MAP aux points de la grille x_t times t -> fonction Omega estimée par krigeage contraint


Z = reshape(Y, n_t, n_x); % gives n_t time n_x matrix of modified put price Z = omega

vol_square = zeros(n_t-1, n_x-2); 

for i = 1:(n_t-1)
    for j = 2:(n_x-1)
        diff_T_Omega = (Z(i+1,j)-Z(i,j))/step_size_t; %estimation par difference finie de la derivee de Omega par rapport a T
        diff_2_k_Omega = (Z(i,j+1)-2*Z(i,j)+Z(i,j-1))/step_size_x^2; %estimation par difference finie de la derivee seconde de Omega par rapport a K modifié
        vol_square(i,j-1) = 2*diff_T_Omega/((x_k(j)^2)*diff_2_k_Omega);
    end
end

vol_Dupire = sqrt(vol_square);

figure; % plot vol surface - strike first
t_new = t(1:(n_t-1));

% Previous version without convertion to unmodified strikes
% x_new = x(2:(n_x-1)); % on utilise la grille des K non modifié pour avoir une representation de la fonction \sigma(T,K) et non \sigma(T,k)
% [xx_new, tt_new] = meshgrid(x_new,t_new);

x_new = x_k(2:(n_x-1)); %grill des K modifié
[xx_new, tt_new] = meshgrid(x_new,t_new); % meshgrid with modified strike
div_int_vect = div_int_func(tt_new);
riskFree_int_vect = riskFree_int_func(tt_new);
xx_new = exp(riskFree_int_vect-div_int_vect).*xx_new; %convertion to unmodifed strikes

surf(xx_new, tt_new, vol_Dupire);
axis tight; grid on;
title('Local Volatility Surface','Fontsize',16,'FontWeight','Bold','interpreter','latex');
xlabel('Strike','Fontsize',16,'FontWeight','Bold','interpreter','latex');
ylabel('Maturity','Fontsize',16,'FontWeight','Bold','interpreter','latex');
zlabel('Local volatility','Fontsize',16,'FontWeight','Bold','interpreter','latex');
set(gca,'Fontsize',16,'LineWidth',1);


%% Plot Black-Scholes vol surface
%x = linspace(min(available_strikes), max(available_strikes), n_x);
%t = linspace(min(available_maturities), max(available_maturities), n_t);
x_vol = linspace(min(available_strikes), max(available_strikes), n_x);  % be careful -> the same n_x than for Dupire vol surface
%x_vol = linspace(min(Strike_test), max(Strike_test), n_x);
t_vol = linspace(min(available_maturities), max(available_maturities), n_t);
% x = linspace(x_nodes.min, x_nodes.max, 50);
% t = linspace(t_nodes.min, t_nodes.max, 50);
[xx_vol, tt_vol] = meshgrid(x_vol,t_vol);
n_x = length(x_vol);
n_t = length(t_vol);
% Compute the finite-dimensional Gaussian process for points in x and t
[Phi_x, Phi_t] = Basis_func_decomp(x_vol, t_vol, x_nodes, t_nodes);
Phi_xt = kron(Phi_x,Phi_t);
Y = Phi_xt*Xi_mode;
figure; % plot vol surface - strike first
Strike_vol = reshape(xx_vol, n_t*n_x,1);
Maturity_vol = reshape(tt_vol, n_t*n_x,1);
Put_price = reshape(Y, n_t*n_x,1);
% call-put parity formula
%data.r = 0.01;
% Compute call price from call-put parity
%Call_price_vol = put_price + data.S0 - exp(-data.r.*(Maturity_vol)).*Strike_vol;
%filter low maturities
Maturity_threshold = 0.18;
index_Maturity_vol = (Maturity_vol >= Maturity_threshold);

r_int_vect = riskFree_int_func(Maturity_vol(index_Maturity_vol));
div_int_vect = div_int_func(Maturity_vol(index_Maturity_vol));

% ZERO Interest rate - ZERO Dividend rate
%r_int_vect = zeros(size(Maturity_vol(index_Maturity_vol)));
%div_int_vect = zeros(size(Maturity_vol(index_Maturity_vol)));

Put_price_vect = exp(-div_int_vect).*Put_price(index_Maturity_vol);

VolSurface_from_putPrice_with_div(data.S0, r_int_vect, div_int_vect, Strike_vol(index_Maturity_vol), Maturity_vol(index_Maturity_vol), Put_price_vect);


%% 8. Sampling of the truncated Gaussian process by exact HMC method -> see Pakman - Paninski paper

% Nb_simu_HCM = 100;
% 
% tic
% 
% % Require sampling of \Xi given \Phi \xi + \tilde{\epsilon} = y which
% % follows a Normal distribution with mean :
% K = data.Phi*Gamma*data.Phi'; 
% [K_inv, det_K] = invChol_mex_2(K+Sigma_noise);
% aux = Gamma*data.Phi';
% mu_cond = aux*K_inv*Beq;
% % as to give the same result as 
% % mu_cond_2 = aux*K_inv*Price_vector;
% 
% % and convariance matrix :
% %nugget = 1e-4;
% %nugget = 5e-4;
% nugget = 1e-3;
% Sigma_cond = Gamma - aux*K_inv*aux';
% Sigma_cond = (Sigma_cond+Sigma_cond')/2 + nugget*eye(data.nb_nodes);
% 
% initial_X = Xi_mode;
% %Nb_simu_HCM = 100;
% 
% tol = 0.000001;
% g = -tol*ones(data.Nbconstr, 1);
% [Xs, bounce_count] = HMC_exact(A, g, Sigma_cond, mu_cond, 'true', Nb_simu_HCM, initial_X);
% 
% save('Xs_100_sample_paths_alloc_9.mat', 'Xs')
% 
% toc


