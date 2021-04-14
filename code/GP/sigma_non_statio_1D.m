function sigma = sigma_non_statio_1D(x, data)

    input = data.T; %assume input is an increasing vector
    input_min = input(1);
    input_max = input(end);
    response = data.callPrice; %assume response is an increasing vector
    response_min = response(1);
    response_max = response(end);
    response_range = response_max-response_min;
    
    if x>input_min && x<=input_max
        [M, index] = min(abs(input-x)); %find the index of the closest element in input
        if x<=input(index) %  input(index-1)< x <= input(index)
            %s_temp = 1;
            %s_temp = response(index)-response(index-1);
            s_temp = (response(index)-response(index-1))/response_range;
        else %  input(index)< x < input(index+1)
            %s_temp = 1;
            %s_temp = response(index+1)-response(index); 
            s_temp = (response(index+1)-response(index))/response_range;
        end 
    end
    
    if x <= input_min
        s_temp = 1;
        %s_temp = (response(2)-response(1));
        %s_temp = (response(2)-response(1))/response_range;

    end
        
    if x > input_max
        s_temp = 1;
        %s_temp = (response(end)-response(end-1));
        %s_temp = (response(end)-response(end-1))/response_range;

    end    

    % ver 1 :
    sigma = data.sigma* s_temp;

end