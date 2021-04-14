function H_vect = basis_func_vect_2(x, t, data)

    x_nodes = data.x_nodes;
    t_nodes = data.t_nodes;
    x_nodes_min = data.x_nodes.vect(1);
    x_nodes_max = data.x_nodes.vect(end);
    t_nodes_min = data.t_nodes.vect(1);
    t_nodes_max = data.t_nodes.vect(end);
    
    %Ntot = data.nb_nodes;

    %H_vect = zeros(1, Ntot);
    
    G = zeros(1, x_nodes.nb);
    H = zeros(1, t_nodes.nb);
    
    %x closest element in x_nodes
    
    if x>x_nodes_min && x<x_nodes_max
        [M, index] = min(abs(x_nodes.vect-x)); %find the index of the closest element in x_nodes
        if x<x_nodes.vect(index) %  x_nodes(index-1)< x <= x_nodes(index)
            G(1,index-1) = 1 - (x-x_nodes.vect(index-1))/x_nodes.delta(index-1);
            G(1,index) = 1 - (x_nodes.vect(index)-x)/x_nodes.delta(index-1);
        else %  x_nodes(index)<= x < x_nodes(index+1)
            G(1,index) = 1 - (x-x_nodes.vect(index))/x_nodes.delta(index);
            G(1,index+1) = 1 - (x_nodes.vect(index+1)-x)/x_nodes.delta(index);  
        end 
    end
    
    if x == x_nodes_min
        G(1,1) = 1;
    end
        
    if x == x_nodes_max
        G(1,x_nodes.nb) = 1;
    end    

    if t>t_nodes_min && t<t_nodes_max
        [M, index] = min(abs(t_nodes.vect-t)); %find the index of the closest element in x_nodes
        if t<=t_nodes.vect(index) %  t_nodes(index-1)< t <= t_nodes(index)
            H(1,index-1) = 1 - (t-t_nodes.vect(index-1))/t_nodes.delta(index-1);
            H(1,index) = 1 - (t_nodes.vect(index)-t)/t_nodes.delta(index-1);
        else %  t_nodes(index) <= t < t_nodes(index+1)
            H(1,index) = 1 - (t-t_nodes.vect(index))/t_nodes.delta(index);
            H(1,index+1) = 1 - (t_nodes.vect(index+1)-t)/t_nodes.delta(index);  
        end
        
    end
    
    if t == t_nodes_min
        H(1,1) = 1;
    end
        
    if t == t_nodes_max
        H(1,t_nodes.nb) = 1;
    end    
    
    
    [GG, HH] = meshgrid(G,H);
    H_vect =  reshape(GG.*HH, 1, data.nb_nodes);    
    
end