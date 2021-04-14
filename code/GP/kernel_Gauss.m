function y = kernel_Gauss(p1, p2, data)
%Input : p1 = [x1, t1], p2 = [x2, t2]
    x1 = p1(1);
    t1 = p1(2);
    x2 = p2(1);
    t2 = p2(2);
    %part1 = exp(-0.5*((x1-x2)/data.theta_x)^2)
    %part2 = exp(-0.5*((t1-t2)/data.theta_t)^2)
    y = (data.sigma^2)*exp(-0.5*((x1-x2)/data.theta_x)^2)*exp(-0.5*((t1-t2)/data.theta_t)^2);
end