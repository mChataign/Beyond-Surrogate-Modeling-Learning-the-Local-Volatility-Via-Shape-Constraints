function y = kernel_Exp_non_statio_1D(p1, p2, data)
%Input : p1 = [x1, t1], p2 = [x2, t2]
    x1 = p1(1);
    x2 = p2(1);
    %theta = data.theta_x;
    theta = data.theta_t;
    sigma = data.sigma;
    
    part1 = sigma_non_statio_1D(x1, data)*sigma_non_statio_1D(x2, data);
    
    l1 = theta*sigma/sigma_non_statio_1D(x1, data);
    l2 = theta*sigma/sigma_non_statio_1D(x2, data);
    part2 = sqrt(2*l1*l2/(l1^2+l2^2));  
    part3 = exp(-(x1-x2)^2/(l1^2+l2^2));
    
   
    
% % %     theta = data.theta_t;
% % %     part2 = (1 + sqrt(5)*abs(t1-t2)/theta + (5/3)*((t1-t2)/theta)^2) * exp(-sqrt(5)*abs(t1-t2)/theta);
    
    y =  part1*part2*part3;
end