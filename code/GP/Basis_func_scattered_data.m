function Phi = Basis_func_scattered_data(X, Y, x_nodes, t_nodes)
    % Construct Phi the (n, N) matrix of basis functions valued at the n points 
    % (X, Y)
    % where the x-coordinates given by a nx1 vector X  
    % with the y-coordinates given by a nx1 vector Y
    %
    % Input : n test points or input locations caracterized by 
    % X : (n, 1) vector of x-axis coordinates
    % Y : (n, 1) vector of y-axis coordinates
    % REMARK : X and Y values should be inside [min-max] values of x_nodes and t_nodes
    % Output 
    % Phi : the (n, N) matrix of basis functions valued at the n points
    
    % X and Y should avec the same size
    n = length(X); % number of input points to assess 
    N = x_nodes.nb*t_nodes.nb; % number of points in the basis function grid (size of vector \xi)
    Phi = zeros(n, N);
    
    for i = 1:n
         Phi(i,:) = basis_func_vect(X(i), Y(i), x_nodes, t_nodes);
    end   
        
    
end