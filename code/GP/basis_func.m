function y = basis_func(x, k, z_nodes)

    
   y = 0;
    
   delta = z_nodes.delta;
   nodes = z_nodes.vect;
   Nb_nodes = z_nodes.nb;
   
    
   if (k == 1)
       if x < nodes(2)
           y = 1 - (x-nodes(1))/delta(1);
       end
       return;
   end

   if (k == Nb_nodes)
       if x > nodes(Nb_nodes-1)
           y = 1 - (nodes(Nb_nodes)-x)/delta(Nb_nodes-1);
       end
       return;
   end
   
   % other cases
   if x > nodes(k-1) && x <= nodes(k)
       y = 1 - (nodes(k)-x)/delta(k-1);
       return;
   end
   
   if x > nodes(k) && x < nodes(k+1)
       y = 1 - (x-nodes(k))/delta(k);
       return;
   end 
   
       
end