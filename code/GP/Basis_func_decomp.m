function [Phi1, Phi2] = Basis_func_decomp(X, Y, x_nodes, t_nodes)
    % Construct Phi1 and Phi2 such that 
    % Phi = (Phi1 \Kronecker_Prod Phi2) is the (n, N) matrix of basis
    % functions valued at the n points (X,Y) 
    % where Phi1(i,j) = \Phi_j(x_i) and Phi2(i,j) = Psi_j(t_i)
    % where n = n1*n2 and N = N1*N2
    %
    % Input : grid of test points or input locations caracterized by 
    % X : (n1, 1) vector of x-axis values in the grid
    % Y : (n2, 1) vector of y-axis values in the grid
    % data   : structure with specification of basis function nodes in all directions
    %           data.x_nodes; data.t_nodes;
    % REMARK : X and Y values should be inside [min-max] values of x_nodes and t_nodes
    % Output 
    % Phi1 : (n1, N1) matrix where n1 = size(X) and N1 = size(data.x_nodes)
    % Phi2 : (n2, N2) matrix where n2 = size(Y) and N2 = size(data.t_nodes)
    
    %x_nodes = data.x_nodes;
    %t_nodes = data.t_nodes;
    
    n1 = length(X);
    n2 = length(Y);
    
    Phi1 = zeros(n1, x_nodes.nb);
    Phi2 = zeros(n2, t_nodes.nb);
    
    for i = 1:n1
        for j=1:x_nodes.nb
            Phi1(i,j) = basis_func(X(i), j, x_nodes); 
        end
    end   
        
    for i = 1:n2
        for j=1:t_nodes.nb
            Phi2(i,j) = basis_func(Y(i), j, t_nodes); 
        end
    end       
    
end