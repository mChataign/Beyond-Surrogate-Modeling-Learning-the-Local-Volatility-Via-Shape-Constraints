function [T, M, IV] = VolSurface_from_putPrice(S0, r, K, T, PutPrice)
%**************************************************************************
% VolSurface - Volatility Surface
%   Compute the implied volatility of an underlying asset from the market 
%   values of European calls and plot the volatility surface.
%
%   surface = VolSurface_from_putPrice(S0, r, K, T, PutPrice)
%
%==========================================================================
% INPUTS: 
%
%   S0      - Current price of the underlying asset.
%
%   r       - Annualized continuously compounded risk-free rate of return over
%           the life of the option, expressed as a positive decimal number.
%
%   K       - Vector of strike (i.e., exercise) prices of the option.
%
%   T       - Vector of times to expiration of the option, expressed in years.
%
%
%   PutPrice   - Vector of prices (i.e., value) of European put options 
%                 from which the implied volatilities of the underlying asset 
%                 are derived.
%
%   Note: the inputs T, K and PutPrice must have the same length N. 
%   They form a set of N R^3 Vectors.
%   i.e. size([Strike, Maturity PutPrice])= N x 3
%
%==========================================================================
% OUTPUTS:
%
%   Surface structure - Matrix of implied volatility of the underlying asset 
%   derived from European option prices.
%   
%   Surface plot      - 3D Implied volatility plot wtr moneyness and time to
%                       maturity
%      
%==========================================================================
% EXAMPLE:
%
%       see: Example1.m 
%            Example2.m         
%
%**************************************************************************
% Rodolphe Sitter - MSFM Student - The University of Chicago
% March 17, 2009
%**************************************************************************


% COMPUTE THE IMPLIED VOLATILITIES
num = length(PutPrice);
ImpliedVol = nan(num, 1);
options = optimset('fzero');
options = optimset(options, 'TolX', 1e-6, 'Display', 'off');
for i = 1:length(ImpliedVol)
    try
          ImpliedVol(i) = fzero(@objfcn, [0 10], options, ...
                                 S0, K(i), T(i), r, PutPrice(i));
                catch
          ImpliedVol(i) = NaN;
    end 
end


% CLEAN MISSING VALUES
%T
M=K./S0;              % moneyness for put option (it is the inverse for call option)
IV=ImpliedVol;       % Implied Volatility
T=T(:); M=M(:); IV=IV(:);
missing=(T~=T)|(M~=M)|(IV~=IV);
T(missing)=[];
M(missing)=[];
IV(missing)=[];

index_zero = (IV == 0);
T(index_zero)=[];
M(index_zero)=[];
IV(index_zero)=[];

%T
%M
%IV


min_M = min(M);
max_M = max(M);
min_T = min(T);
max_T = max(T); 
M_lin = linspace(min_M, max_M, 100);
T_lin = linspace(min_T, max_T, 100);

[MM, TT] = meshgrid(M_lin, T_lin);
Implied_vol_mat = griddata(M,T,IV,MM,TT, 'cubic');

% % CHOOSE BANDWIDTH hT and hM
% hT=median(abs(T-median(T)));    surface.hT=hT;
% hM=median(abs(M-median(M)));    surface.hM=hM;
% % CHOOSE GRID STEP N 
% TT = sort(T);     MM = sort(M);
% NT = histc(T,TT); NM = histc(M,MM);
% NT(NT==0) = [];   NM(NM==0) = [];
% nt=length(NT);    nm=length(NM);
% N=min(max(nt,nm),70);
% 
% 
% % SMOOTHE WITH GAUSSIAN KERNEL 
% kerf=@(z)exp(-z.*z/2)/sqrt(2*pi);
% surface.T=linspace(min(T),max(T),N);
% surface.M=linspace(min(M),max(M),N);
% surface.IV=nan(1,N);
% for i=1:N
%     for j=1:N
%     z=kerf((surface.T(j)-T)/hT).*kerf((surface.M(i)-M)/hM); 
%     surface.IV(i,j)=sum(z.*IV)/sum(z);
%     end
% end

% Without smoothing with Gaussian Kernel
% surface.T=linspace(min(T),max(T),N);
% surface.M=linspace(min(M),max(M),N);
% surface.IV=IV;

% linspace(min(T),max(T),N)
% linspace(min(M),max(M),N)
% IV

% PLOT THE VOLATILITY SURFACE
surf(MM,TT,Implied_vol_mat)
axis tight; grid on;
title('Implied Volatility Surface','Fontsize',14,'FontWeight','Bold','interpreter','latex');
xlabel('Put Moneyness $M=\frac{K}{S}$','Fontsize',14,'FontWeight','Bold','interpreter','latex');
ylabel('Time to Matutity $T$','Fontsize',14,'FontWeight','Bold','interpreter','latex');
zlabel('Implied Volatility $\sigma(M,T)$','Fontsize',14,'FontWeight','Bold','interpreter','latex');
set(gca,'Fontsize',14,'FontWeight','Bold','LineWidth',1);

end
%==========================================================================
% BLACK-SCHOLES PRICE

function P = BlackScholesPrice_Put(S0,K,T,r,sigma)

    d1=(log(S0./K)+(r+0.5.*sigma.^2)*T)./(sigma.*sqrt(T));
    d2=(log(S0./K)+(r-0.5.*sigma.^2)*T)./(sigma.*sqrt(T));

    P = K.*exp(-r.*T).*normcdf(-d2)-S0.*normcdf(-d1);
    
end

% function C = BlackScholesPrice_Call(S0,K,T,r,sigma)
% 
%     d1=(log(S0./K)+(r+0.5.*sigma.^2)*T)./(sigma.*sqrt(T));
%     d2=(log(S0./K)+(r-0.5.*sigma.^2)*T)./(sigma.*sqrt(T));
% 
%     C = S0.*normcdf(d1) - K.*exp(-r.*T).*normcdf(d2);
%     
% end


% function BlackScholesPrice=BlackScholesPricer(S0,K,T,r,sigma)
% 
% F=S0.*exp(r.*T);
% d1=log(F./K)./(sigma.*sqrt(T))+sigma.*sqrt(T)/2;
% d2=log(F./K)./(sigma.*sqrt(T))-sigma.*sqrt(T)/2;
% BlackScholesPrice = exp(-r.*T).*(F.*normcdf(d1)-K.*normcdf(d2));


%==========================================================================
% BLACK-SCHOLES IMPLIED VOLATILITY OBJECTIVE FUNCTION
function delta = objfcn(volatility, S0, K, T, r, PutPrice)

    BSprice = BlackScholesPrice_Put(S0, K, T, r, volatility);
    delta = PutPrice - BSprice;

end
%==========================================================================
