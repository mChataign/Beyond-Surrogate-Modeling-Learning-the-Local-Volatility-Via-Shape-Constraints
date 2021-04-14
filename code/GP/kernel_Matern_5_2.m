function y = kernel_Matern_5_2(p1, p2, data)
%Input : p1 = [x1, t1], p2 = [x2, t2]
    x1 = p1(1);
    t1 = p1(2);
    x2 = p2(1);
    t2 = p2(2);
    theta = data.theta_x;
    part1 = (1 + sqrt(5)*abs(x1-x2)/theta + (5/3)*((x1-x2)/theta)^2) * exp(-sqrt(5)*abs(x1-x2)/theta);
    
    theta = data.theta_t;
    part2 = (1 + sqrt(5)*abs(t1-t2)/theta + (5/3)*((t1-t2)/theta)^2) * exp(-sqrt(5)*abs(t1-t2)/theta);
    
    y = (data.sigma^2)*part1*part2;
end