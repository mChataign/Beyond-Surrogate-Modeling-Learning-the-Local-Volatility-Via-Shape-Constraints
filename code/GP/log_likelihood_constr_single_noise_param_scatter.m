function Log_like = log_likelihood_constr_single_noise_param_scatter(param, y, data)
    % Construct the log-likelihood in the contrained case : 
    % log P(Y^N = y \given param) + log P(AY^N >= 0 \given Phi \xi +
    % \tilde{epsilon} = y,   param) - log P(AY^N >= 0 \given   param) 
    % It is assumed that observations are gridded data -> Matrix of basis
    % function Phi have a Kronecker decomposition Phi_1 Phi2 and data.Phi_1
    % and data.Phi2 are available
    %
    % Input : vector of hyper-parameters
    % param(1) = sigma
    % param(2) = theta_1
    % param(3) = theta_2
    % param(4) = sigma_noise   
    % y : market price obervation vector
    % data   : structure with specification of basis function nodes in all directions
    %           data.x_nodes; data.t_nodes;
    % Log_like : log-likelihood for truncated Gaussian vector
     
    sigma = param(1);
    theta_x = param(2);
    theta_t = param(3);
    % Kronecker decomposition of covariance function
    [Gamma1, Gamma2] = Gamma_decomp(sigma, theta_x, theta_t, data);
    Gamma = kron(Gamma1, Gamma2);
    Gamma = (Gamma + Gamma')/2;
    
    noise = param(4);
    Sigma_noise = noise*eye(data.nb_price);
    K = data.Phi * Gamma * data.Phi';
    [K_inv, det_K] = invChol_mex_2(K+Sigma_noise);
    
    %[K1_inv, det_K1] = invChol_mex_2(K1_with_noise);
    %[K2_inv, det_K2] = invChol_mex_2(K2_with_noise);

    %det_K = (det_K1^(data.nb_K))*(det_K2^(data.nb_T)); %det of Kronecker matrix product
    %K_inv = kron(K1_inv, K2_inv); %inverse of Kronecker matrix product

    %% Unconstrained log-likelihood :
    Log_like_unconstr = -log(det_K) - 0.5*y'*K_inv*y;
    
    
    %% Construct part 2 in the constrained log-likelihood : 
    % P(AY^N >= 0 \given Phi \xi + \tilde{epsilon} = y,   param) 
    % where A is the inequality constraint matrix
    % It is assumed that observations are gridded data -> Matrix of basis
    % function Phi have a Kronecker decomposition Phi_1 Phi2 and data.Phi_1
    % and data.Phi2 are available

    % AY^N given Phi \xi + \tilde{epsilon} = y is a Gaussian vector with mean
    % and covariance such that :
    aux = data.A * Gamma * data.Phi';
    mean_cond = aux* K_inv * y;
    cov_AY = data.A * Gamma * data.A';
    cov_cond = cov_AY - aux* K_inv * aux';
    nugget = 10^(-5)*eye(data.Nbconstr);
    cov_cond = (cov_cond+cov_cond')/2 + nugget;

    % Dimension reduction : identification of active directions -> see paper by Azzimonti -
    % Ginsbourger
    W = normcdf(mean_cond./sqrt(diag(cov_cond))); %criteria, dimension i is selected if W_i is low enough
    threshold = 0.95;
    % first approach : deterministic selection of indices
    index1 = find(W < threshold);
    Nb_index = length(index1);
    % first approach : sampling active directions -> see paper by Azzimonti -
    % Ginsbourger
    %s = RandStream('mlfg6331_64');
    %d = 100;
    %index = datasample(s,1:Nbconstr,d,'Replace',false,'Weights',W);

    % Computation of cond mean and cov on the subset of indices
    New_cov_cond = cov_cond(index1,index1);
    New_mean_cond = mean_cond(index1);

    lower_point = zeros(Nb_index,1);
    upper_point = inf*ones(Nb_index,1);

    %res = mvncdf(lower_point,upper_point,New_mean_cond,New_cov_cond); -> do
    %not work for dimension larger than 25
    % Compute upper-orthant proba : \P(Z > 0) with Z Gaussian with
    % New_mean_cond and New_cov_cond
    Like_constr_part_1 = mvtcdfqmc(lower_point-New_mean_cond, upper_point, corrcov(New_cov_cond), Inf)
    %[p, err, funevals] = mvtcdfqmc(lower_point-New_mean_cond, upper_point, corrcov(New_cov_cond), Inf);
    
    if Like_constr_part_1 == 0
        Log_like_constr_part_1 = -10^8;
    else
        Log_like_constr_part_1 = log(Like_constr_part_1);
    end


    %% Construct part 3 in the constrained log-likelihood : 
    % P(AY^N >= 0,   param) 
    % where A is the inequality constraint matrix
    cov_AY = (cov_AY+cov_AY')/2 + nugget;

    % Dimension reduction : identification of active directions -> see paper by Azzimonti -
    % Ginsbourger
    % W = normcdf(mean_cond./sqrt(diag(cov_cond))); %criteria, dimension i is selected if W_i is low enough
    % threshold = 0.95;
    % % first approach : deterministic selection of indices
    % index1 = find(W < threshold);
    % Nb_index = length(index1);
    % % first approach : sampling active directions -> see paper by Azzimonti -
    % % Ginsbourger
    % %s = RandStream('mlfg6331_64');
    % %d = 100;
    % %index = datasample(s,1:Nbconstr,d,'Replace',false,'Weights',W);
    % 
    % % Computation of cond mean and cov on the subset of indices
    % New_cov_cond = cov_cond(index1,index1);
    % New_mean_cond = mean_cond(index1);
    % 
    % lower_point = zeros(Nb_index,1);
    % upper_point = inf*ones(Nb_index,1);

    % without dimension reduction :
    lower_point = zeros(data.Nbconstr,1);
    upper_point = inf*ones(data.Nbconstr,1);
    Like_constr_part_2 = mvtcdfqmc(lower_point, upper_point, corrcov(cov_AY), Inf)
    
    if Like_constr_part_2 == 0
        Log_like_constr_part_2 = -10^8;
    else
        Log_like_constr_part_2 = log(Like_constr_part_2);
    end
    
    %% Sum up all parts
    Log_like = Log_like_unconstr + Log_like_constr_part_1 - Log_like_constr_part_2;
    
end