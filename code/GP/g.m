function y = g(x, i, data)

    x_nodes = data.x_nodes;
    x_nodes_vect = x_nodes.vect;
    x_nodes_delta = x_nodes.delta;

    if x<x_nodes.min || x>x_nodes.max
        y=0;
        return;
    end    
    
    if i==1 %x(i)= x_0 first node
        x_i = x_nodes_vect(1);
        delta_i = x_nodes_delta(1);
        y = max(0, 1-(x-x_i)/delta_i);
        return;
    end
    
    if i== x_nodes.nb %x(i)= x_{N_x} last note
        x_i = x_nodes_vect(end);
        delta_i = x_nodes_delta(end);
        y = max(0, 1-(x_i-x)/delta_i);
        return;
    end
    

   x_i = x_nodes_vect(i);
   delta_i_minus_1 = x_nodes_delta(i-1);
   delta_i = x_nodes_delta(i);
   if x< x_i
       y = max(0, 1-(x_i-x)/delta_i_minus_1);
       return 
   else 
       y = max(0, 1-(x-x_i)/delta_i);
   end
           
    
end