function y = h(t, j, data)

    t_nodes = data.t_nodes;
    t_nodes_vect = t_nodes.vect;
    t_nodes_delta = t_nodes.delta;
    
    if t<t_nodes.min || t>t_nodes.max
        y=0;
        return;
    end
    
    if j==1 %x(i)= x_0 first node
        t_j = t_nodes_vect(1);
        delta_j = t_nodes_delta(1);
        y = max(0, 1-(t-t_j)/delta_j);
        return;
    end
    
    if j== data.t_nodes.nb %x(i)= x_{N_x} last note
        t_j = t_nodes_vect(end);
        delta_j = t_nodes_delta(end);
        y = max(0, 1-(t_j-t)/delta_j);
        return;
    end
    
   t_j = t_nodes_vect(j);
   delta_j_minus_1 = t_nodes_delta(j-1);
   delta_j = t_nodes_delta(j);
   if t< t_j
       y = max(0, 1-(t_j-t)/delta_j_minus_1);
       return 
   else 
       y = max(0, 1-(t-t_j)/delta_j);
   end
           
    
end