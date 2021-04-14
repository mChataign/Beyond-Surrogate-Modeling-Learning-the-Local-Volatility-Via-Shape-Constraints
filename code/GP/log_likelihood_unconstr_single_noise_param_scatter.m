function Log_like = log_likelihood_unconstr_single_noise_param_scatter(param, y, data)
    % Construct the log-likelihood in the uncontrained case : P(Y^N = y \given param) 
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
    % Log_like : log-likelihood in the Gaussian case without constant term
     
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
    %det_K = det(K+Sigma_noise);
    %K_inv = inv(K+Sigma_noise);
    
    %[K1_inv, det_K1] = invChol_mex_2(K1_with_noise);
    %[K2_inv, det_K2] = invChol_mex_2(K2_with_noise);

    %det_K = (det_K1^(data.nb_K))*(det_K2^(data.nb_T)); %det of Kronecker matrix product
    %K_inv = kron(K1_inv, K2_inv); %inverse of Kronecker matrix product

    %y = reshape(data.callPrice', data.nb_price, 1);

    %det = det_K
    
    Log_like = -log(det_K) - 0.5*y'*K_inv*y;
    
end