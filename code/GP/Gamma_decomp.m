function [Gamma1, Gamma2] = Gamma_decomp(sigma, theta_x, theta_t, data)
    % Construct Gamma1 and Gamma2 such that 
    % Gamma = cov(x_nodes, t_nodes) = sigma^2 (Gamma1 \Kronecker_Prod Gamma2)
    % where Gamma1 = cov(x_nodes) and Gamma2 = cov(t_nodes)
    % REMARK : no need of sigma parameter in this function
    % sigma will be used just after calling Gamma_decomp
    %
    % Input : 
    % hyper_param : [theta_x, theta_t]
    % data   : structure with specification of basis function nodes in all directions
    %           data.x_nodes; data.t_nodes;
    % Output 
    % Gamma1 : (N1, N1) covariance matrix where N1 = size(data.x_nodes)
    % Gamma2 : (N2, N2) covariance matrix where N2 = size(data.t_nodes)    
    
    x_nodes = data.x_nodes_scaled;
    t_nodes = data.t_nodes_scaled;    
    %theta_x = data.theta_x;
    %theta_t = data.theta_t;
    
    Gamma1 = zeros(x_nodes.nb, x_nodes.nb);
    Gamma2 = zeros(t_nodes.nb, t_nodes.nb);
    
    for i = 1:x_nodes.nb
        for j=1:x_nodes.nb
            Gamma1(i,j) = kernel_Matern_5_2_1D(x_nodes.vect(i), x_nodes.vect(j), theta_x);
            %Gamma1(i,j) = kernel_Gauss_1D(x_nodes.vect(i), x_nodes.vect(j), theta_x);            
        end
    end   
        
    for i = 1:t_nodes.nb
        for j=1:t_nodes.nb
            Gamma2(i,j) = kernel_Matern_5_2_1D(t_nodes.vect(i), t_nodes.vect(j), theta_t);
            %Gamma2(i,j) = kernel_Gauss_1D(t_nodes.vect(i), t_nodes.vect(j), theta_t);
        end
    end       
    
    Gamma1 = sigma * Gamma1;
    Gamma2 = sigma * Gamma2;
end