function H_vect = basis_func_vect_1D(x, data)

    %x_nodes = data.x_nodes;
    x_nodes = data.t_nodes;

    
    G = zeros(1, x_nodes.nb);
        
    
    if x>x_nodes.min && x<x_nodes.max
        [M, index] = min(abs(x_nodes.vect-x)); %find the index of the closest element in x_nodes
        if x<x_nodes.vect(index) %  x_nodes(index-1)< x <= x_nodes(index)
            G(1,index-1) = 1 - (x-x_nodes.vect(index-1))/x_nodes.delta(index-1);
            G(1,index) = 1 - (x_nodes.vect(index)-x)/x_nodes.delta(index-1);
        else %  x_nodes(index)<= x < x_nodes(index+1)
            G(1,index) = 1 - (x-x_nodes.vect(index))/x_nodes.delta(index);
            G(1,index+1) = 1 - (x_nodes.vect(index+1)-x)/x_nodes.delta(index);  
        end 
    end
    
    if x == x_nodes.min
        G(1,1) = 1;
    end
        
    if x == x_nodes.max
        G(1,x_nodes.nb) = 1;
    end    


    
% % %     [GG, HH] = meshgrid(G,H);
% % %     H_vect =  reshape(GG.*HH, 1, data.nb_nodes);    

    H_vect = G;
    
end