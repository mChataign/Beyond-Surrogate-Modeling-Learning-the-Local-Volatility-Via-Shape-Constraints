function Log_like = log_likelihood_unconstr_two_noise_param(param, data)
    % Construct the log-likelihood in the uncontrained case : P(Y^N = y \given param) 
    % It is assumed that observations are gridded data -> Matrix of basis
    % function Phi have a Kronecker decomposition Phi_1 Phi2 and data.Phi_1
    % and data.Phi2 are available
    %
    % Input : vector of hyper-parameters
    % param(1) = sigma
    % param(2) = theta_1
    % param(3) = theta_2
    % param(4) = sigma_noise_1    
    % param(5) = sigma_noise_2   
    % data   : structure with specification of basis function nodes in all directions
    %           data.x_nodes; data.t_nodes;
    % y : log-likelihood in the Gaussian case without constant term
     
    sigma = param(1);
    theta_x = param(2);
    theta_t = param(3);
    % Kronecker decomposition of covariance function
    [Gamma1, Gamma2] = Gamma_decomp(sigma, theta_x, theta_t, data);
    
    sigma_noise_1 = param(4);
    sigma_noise_2 = param(5);
    Sigma_noise1 = sigma_noise_1*eye(data.nb_K);
    Sigma_noise2 = sigma_noise_2*eye(data.nb_T);

    K1 = data.Phi1*Gamma1*data.Phi1'; 
    K1_with_noise = (K1 + K1')/2 + Sigma_noise1;
    K2 = data.Phi2*Gamma2*data.Phi2'; 
    K2_with_noise = (K2 + K2')/2 + Sigma_noise2;
    %K = kron(K1, K2);

    [K1_inv, det_K1] = invChol_mex_2(K1_with_noise);
    [K2_inv, det_K2] = invChol_mex_2(K2_with_noise);

    det_K = (det_K1^(data.nb_K))*(det_K2^(data.nb_T)); %det of Kronecker matrix product
    K_inv = kron(K1_inv, K2_inv); %inverse of Kronecker matrix product

    y = reshape(data.callPrice', data.nb_price, 1);

    Log_like = -log(det_K) - 0.5*y'*K_inv*y;
    
end