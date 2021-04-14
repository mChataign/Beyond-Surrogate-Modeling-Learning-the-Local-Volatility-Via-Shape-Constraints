%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 		
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    Kriging_non_statio_kernel_1D.m
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  In this program, we study a constrained kriging problem using a
%  non-stationnary a priori GP (non-stationnary kernel) that depends on
%  input data and inegality constraints (observed data and known constraints such as monotonicity or convexity).
%
%  Created on 2018-08-07, at 13:53, by Areski Cousin
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

%% Number of simulation
data.nb_simu = 50;

%%  Inputs data : import call price data using input location data.T, data.K and response data.callPrice
SPX=xlsread('SPX.xls');
data.S0=770.05; % Current price of the underlying asset.
data.r=0.05;  % Annualized continuously compounded risk-free rate of return, expressed as a positive decimal number

Maturity = SPX(3:end,2);          % maturity
Strike = SPX(3:end,3);            % moneyness
CallPrice =SPX(3:end,4);         % option prices

data.T = unique(Maturity)';  % Vector of times to expiration of the option, expressed in years.
nb_T = length(data.T);

%set_K = cell(1, Maturity);
%set_Call = cell(1, Maturity);

set_K = unique(Strike)';
for i=1:nb_T
    Index_Ti = (Maturity == data.T(i));
    set_K = intersect(set_K, Strike(Index_Ti));
end
data.K = set_K';
nb_K = length(data.K);

data.callPrice = zeros(nb_K, nb_T);
for i=1:nb_K
    for j=1:nb_T
        Index_callPrice_ij = (Strike==data.K(i)) & (Maturity == data.T(j));
        data.callPrice(i,j) = mean(CallPrice(Index_callPrice_ij));   
    end
end

%Index_K = [4 10]
%Index_T = [3 7];
%Index_K = 4:nb_K-4; %on enleve les 2 premieres lignes qui ont des arbitrages

Index_T = 3:nb_T;
%Index_T = 3:5;
Index_K = [6]; % we only consider one particular strike price
data.T = data.T(Index_T)
data.K = data.K(Index_K)

data.callPrice = data.callPrice(Index_K, Index_T); 
eps = 1;
data.callPrice(5) = data.callPrice(6) - eps;

data.nb_T = length(data.T);
data.nb_K = length(data.K);
data.nb_price =  data.nb_K*data.nb_T;

scatter(data.T, data.callPrice, 'MarkerEdgeColor','k', 'MarkerFaceColor','k'); %plot callPrice against input location data.T


%% Properties of the (bivariate) Gaussian process
data.sigma = 0.2;
data.theta_t = 0.1;


%% Construction of the grid of basis functions
% Range in maturity time
t_range =  data.T(end) - data.T(1);
t_coef_ext = 0.001;
t_nodes.min = data.T(1)-t_coef_ext*t_range;
t_nodes.max = data.T(end)+t_coef_ext*t_range;
%t_nodes.min = data.T(1);
%t_nodes.max = data.T(end);
t_nodes.nb = 40;
t_nodes.vect = linspace(t_nodes.min, t_nodes.max, t_nodes.nb);
t_nodes.nb = length(t_nodes.vect);
t_nodes.delta = t_nodes.vect(2:end)-t_nodes.vect(1:(end-1));

%data.x_nodes = x_nodes;
data.t_nodes = t_nodes;
data.nb_nodes = t_nodes.nb;

 
%% Construction of covariance matrix Gamma of $\xi$
Gamma = zeros(data.nb_nodes, data.nb_nodes);
k=1;
for j1=1:t_nodes.nb
    l=1;
        for j2=1:t_nodes.nb
            p1 = [t_nodes.vect(j1)];
            p2 = [t_nodes.vect(j2)];                  
            %Gamma(k,l) = kernel_Gauss(p1, p2, data);
            %Gamma(k,l) = kernel_Matern_5_2_1D(p1, p2, data);
            Gamma(k,l) = kernel_Matern_5_2_non_statio_1D(p1, p2, data);
            %Gamma(k,l) = kernel_Exp_non_statio_1D(p1, p2, data);
            %test = Gamma(k,l);
            l=l+1;
        end
    k=k+1;
end   
nugget = 0;
%nugget = 1e-8;
Gamma = Gamma + nugget*eye(data.nb_nodes)
% % size(Gamma)
% % figure;
% % spy(Gamma);

%% Construction of equality constraints
Aeq = [];
Beq = [];
% market fit constraint
for j=1:data.nb_T
   current_row_Aeq = basis_func_vect_1D(data.T(j), data);
   current_row_Beq = data.callPrice(1,j);
   Aeq = [Aeq; current_row_Aeq];  
   Beq = [Beq; current_row_Beq];         
end


data.nb_eq_constr = length(Beq);    
if (data.nb_eq_constr>=data.nb_nodes)
    disp('Problem : the number of nodes should be greater than the number of constraints');
end
   
    
%% Construction of inequality constraints
A = [];
B = [];
% % % % condition 1) convexity in strike
% % % for i=2:x_nodes.nb-1 
% % %     for j=0:t_nodes.nb-1 
% % %        current_row_A = zeros(1,data.nb_nodes);
% % %        current_row_A(1, i*t_nodes.nb+j+1) = 1;
% % %        current_row_A(1, (i-1)*t_nodes.nb+j+1) = -2;
% % %        current_row_A(1, (i-2)*t_nodes.nb+j+1) = 1;
% % %        A = [A; current_row_A];  
% % %        B = [B; 0];         
% % %     end
% % % end
   
% condition 2) increasingness in maturity
    for j=1:t_nodes.nb-1 
       current_row_A = zeros(1,data.nb_nodes);
       current_row_A(1, j+1) = 1;
       current_row_A(1, j) = -1;
       A = [A; current_row_A];  
       B = [B; 0];         
    end
%A
%B

%% Computation of the mode estimator
% % % ### Computation of the mode estimator
% % % ### decomposition de Cholesky
% % % invGamma <- chol2inv(chol(Gamma)) #Inversion of Gamma from Choleski decomposition
% % % Amat1 <- diag(N+2) #Inequality constraints (AOA condition)
% % % Amat1[1, 1] <- 0 #No constraint bears on \eta (see formula 15 and Prop. 4.1)
% % % Amat <- rbind(A, -Amat1) #A et -Amat1 sont concaténés
% % % zetoil <- solve.QP(invGamma, dvec=rep(0, N+2), Amat=t(Amat), bvec=c(rep(1, m), rep(0, N+2)), meq=m)$solution   #Find the mode (under the linear equality and the inequality constraints) as the solution of a quadratic optimisation problem

invGamma = inv(Gamma);
f = zeros(data.nb_nodes,1);
opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','TolCon', 1e-12, 'Display','iter');
% % % opts = optimoptions('quadprog',...
% % %     'Algorithm','active-set','Display','iter');


[Xi_mode, fval] = quadprog(invGamma,f,-A,B,Aeq,Beq, [], [], [], opts);    
Xi_mode
fval

% % % Test Mode

t = linspace(t_nodes.min, t_nodes.max, 50);
n_t = length(t);
HH = zeros(1,n_t);
    for j=1:n_t
        HH(1,j) = basis_func_vect_1D(t(j), data)*Xi_mode;
    end
% % % figure;
% % % surf(xx, tt, HH');
figure;
hold on;
plot(t, HH);
scatter(data.T, data.callPrice, 'MarkerEdgeColor','k', 'MarkerFaceColor','k'); %plot callPrice against input location data.T



%% Simulation of truncated Gaussian vectors


% First step : Dimension reduction - projecteur M-orthogonal sur un sous-espace vectoriel H  - cours O.Taramasco section 2.3.3 p.37 moindre carré pondérés

data.deg_freedom = data.nb_nodes - data.nb_eq_constr;
Id = eye(data.nb_nodes);
BB = Id-Aeq'*inv(Aeq*Aeq')*Aeq;  %Proj ortho sur F(X)^{Ortho} où F(X) est l'espace vectoriel engendré par les lignes de Aeq

% ------ THIS FORMULA HAS BEEN TESTED (see Test_GP_Sampling) -----------
[epsilontilde, D] = eig(BB'*invGamma*BB); %matrice de changement de base
% --------------------------------------------

c = diag(D);
[c_sort,I] = sort(c,'descend');
Index_first_degfree = I(1:data.deg_freedom);
epsilon = BB*epsilontilde(:, Index_first_degfree); %matrice de projection sur l'espace de taille p (F_0), epsilon de taille N+2 * p
d = 1./c(Index_first_degfree); %inverse des p premieres valeurs propres: variance du vecteur gaussien dans l'espace réduit. Ce vecteur a des composantes indépendantes 

%epsilon = BB*epsilontilde(:, 1:data.deg_freedom); %matrice de projection sur l'espace de taille p (F_0), epsilon de taille N+2 * p
%d = 1./c(1:data.deg_freedom); %inverse des p premieres valeurs propres: variance du vecteur gaussien dans l'espace réduit. Ce vecteur a des composantes indépendantes 


zcentre = Gamma*Aeq'*inv(Aeq*Gamma*Aeq')*Beq; %mean of the untruncated gaussian distribution
setoil = epsilon' * (Xi_mode - zcentre); %moyenne du vecteur gaussien à simuler dans l'espace réduit de taille p

% Second step : simulation of the truncated Gaussian vector using the
% algorithm developped in Maatouk and Bay (2014) : "A New Rejection
% Sampling Method for Truncated Multivariate Gaussian Random Variables Restricted to Convex Sets"

Tolerance = 0;

Xi = zeros(data.nb_nodes, data.nb_simu);
%Xi(end-1, :) = ones(1,data.nb_simu);
for j=1:data.nb_simu 
    Nb_simu = j
    Xi_current = Xi(:,j);
    unif = 1;
    t = 0;
    while (unif > t)
        %Xi_current = Xi(:, j);
        s = setoil + sqrt(d).*normrnd(0 ,1, data.deg_freedom, 1); %simulation du vecteur gaussien de taille data.deg_freedom de moyenne setoil et de variance d dans l'espace réduit F_0
        Xi_current = zcentre + (epsilon * s); %- epsilon %*% setoil #Xi_current est le vecteur gaussien dans l'espace d'origine (de taille N+2)
        while (min(A*Xi_current) < -Tolerance)
            s = setoil + sqrt(d).*normrnd(0 ,1, data.deg_freedom, 1); %simulation du vecteur gaussien de taille data.deg_freedom de moyenne setoil et de variance d dans l'espace réduit F_0
            Xi_current = zcentre + (epsilon * s); %- epsilon %*% setoil #Xi_current est le vecteur gaussien dans l'espace d'origine (de taille N+2)
            %Xi_current
        end
        t = exp(sum(setoil .* setoil ./  d) - sum(s .* setoil ./ d)); %ratio des densités pour la simulation rejet : rejection sampling from the model (voir article Maatouk and Bay, "a new rejection sampling method for multivariate Gaussian random variables restricted to convex sets" Theorem 3 p.4)
        unif = rand;
    end
    Xi(:, j) = Xi_current;
end


%% Plot of GP sample paths
t = linspace(t_nodes.min, t_nodes.max, 200);
n_t = length(t);

figure;
for i=1:data.nb_simu
    HH = zeros(n_t, 1);
    for k=1:n_t
        HH(k,1) = basis_func_vect_1D(t(k), data)*Xi(:,i);
    end
    plot(t, HH);
    hold on;
end
scatter(data.T, data.callPrice, 'MarkerEdgeColor','k', 'MarkerFaceColor','k'); %plot callPrice against input location data.T



