function Log_like = log_likelihood_part_2(param, data)
    % Construct part 2 in the constrained log-likelihood : P(AY^N >= 0 \given Phi \xi + \tilde{epsilon} = y,   param) 
    % where A is the inequality constraint matrix
    % It is assumed that observations are gridded data -> Matrix of basis
    % function Phi have a Kronecker decomposition Phi_1 Phi2 and data.Phi_1
    % and data.Phi2 are available
    %
    % Input : vector of hyper-parameters
    % param(1) = sigma
    % param(2) = theta_1
    % param(3) = theta_2
    % param(4) = sigma_noise   
    % data   : structure with specification of basis function nodes in all directions
    %           data.x_nodes; data.t_nodes;
    % y : P(AY^N >= y \given Phi \xi + \tilde{epsilon} = y,   param) 
     
    sigma = param(1);
    theta_x = param(2);
    theta_t = param(3);
    % Kronecker decomposition of covariance function
    [Gamma1, Gamma2] = Gamma_decomp(sigma, theta_x, theta_t, data);
    Gamma = kron(Gamma1, Gamma2);
    Gamma = (Gamma + Gamma')/2;
    
    % to be put outside for offline computation
    Phi = kron(data.Phi1, data.Phi2);
    y = reshape(data.callPrice', data.nb_price, 1);
    
    noise = param(4);
    Sigma_noise = noise*eye(data.nb_K*data.nb_T);
    
    
    K1 = data.Phi1*Gamma1*data.Phi1'; 
    K1 = (K1 + K1')/2;
    K2 = data.Phi2*Gamma2*data.Phi2'; 
    K2 = (K2 + K2')/2;
    K = kron(K1, K2);
    [K_inv, det_K] = invChol_mex_2(K+Sigma_noise);
    
    % AY^N given Phi \xi + \tilde{epsilon} = y is a Gaussian vector with 
    
    aux = data.A* Gamma * Phi';
    mean_cond = aux* K_inv * y;
    cov_cond = data.A * Gamma * data.A' - aux* K_inv * aux';

    
    Nbconstr  = length(data.B);
    upper_point = 100*ones(Nbconstr, 1);
    
    nugget = 10^(-5)*eye(Nbconstr);
    cov_cond = (cov_cond+cov_cond')/2 + nugget;
    W = normcdf(mean_cond./sqrt(diag(cov_cond)));
    
    index1 = find(W < 0.95);
    Nb_index = length(index1);
    
    New_cov_cond = cov_cond(index1,index1);
    New_mean_cond = mean_cond(index1);
    
    %s = RandStream('mlfg6331_64');
    %d = 100;
    %index = datasample(s,1:Nbconstr,d,'Replace',false,'Weights',W);
        
    %Log_like = index1;
    %Log_like = cov_mat;
    
    
    lower_point = zeros(Nb_index,1);
    upper_point = inf*ones(Nb_index,1);
    
    %res = mvncdf(lower_point,upper_point,New_mean_cond,New_cov_cond);
    [p, err, funevals] = mvtcdfqmc(lower_point-New_mean_cond, upper_point, corrcov(New_cov_cond), Inf);
    
    Log_like = p;
    
    
 
end