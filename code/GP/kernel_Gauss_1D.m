function y = kernel_Gauss_1D(x1, x2, theta)
    % Compute the covariance bewteen 1D points x1 and x2, given kernel
    % parameter theta 
    y = exp(-0.5*((x1-x2)/theta)^2);
end