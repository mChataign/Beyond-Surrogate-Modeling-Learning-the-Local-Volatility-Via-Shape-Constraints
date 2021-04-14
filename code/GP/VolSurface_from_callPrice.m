function surface = VolSurface_from_callPrice(S0, r, T, K, CallPrice)
%**************************************************************************
% VolSurface - Volatility Surface
%   Compute the implied volatility of an underlying asset from the market 
%   values of European calls and plot the volatility surface.
%
%   surface = VolSurface(S0, r, T, K, CallPrice)
%
%==========================================================================
% INPUTS: 
%
%   S0      - Current price of the underlying asset.
%
%   r       - Annualized continuously compounded risk-free rate of return over
%           the life of the option, expressed as a positive decimal number.
%
%   T       - Vector of times to expiration of the option, expressed in years.
%
%   K       - Vector of strike (i.e., exercise) prices of the option.
%
%   CallPrice   - Vector of prices (i.e., value) of European call options 
%                 from which the implied volatilities of the underlying asset 
%                 are derived.
%
%   Note: the inputs T, K and Callprice must have the same length N. 
%   They form a set of N R^3 Vectors.
%   i.e. size([Maturity, Strike, CallPrice])= N x 3
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
num = length(CallPrice);
ImpliedVol = nan(num, 1);
options = optimset('fzero');
options = optimset(options, 'TolX', 1e-6, 'Display', 'off');
for i = 1:length(ImpliedVol)
    try
          ImpliedVol(i) = fzero(@objfcn, [0 10], options, ...
                                 S0, K(i), T(i), r, CallPrice(i));
                catch
          ImpliedVol(i) = NaN;
    end 
end


% CLEAN MISSING VALUES
%T
M=S0./K;              % moneyness for call option (it is the inverse for put option)
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

min_T = min(T);
max_T = max(T); 
min_M = min(M);
max_M = max(M);
T_lin = linspace(min_T, max_T, 100);
M_lin = linspace(min_M, max_M, 100);

[TT,MM] = meshgrid(T_lin, M_lin);
Implied_vol_mat = griddata(T,M,IV,TT,MM, 'cubic');

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
surf(TT,MM,Implied_vol_mat)
axis tight; grid on;
title('Implied Volatility Surface','Fontsize',24,'FontWeight','Bold','interpreter','latex');
xlabel('Time to Matutity $T$','Fontsize',20,'FontWeight','Bold','interpreter','latex');
ylabel('Moneyness $M=\frac{S}{K}$','Fontsize',20,'FontWeight','Bold','interpreter','latex');
zlabel('Implied Volatility $\sigma(T,M)$','Fontsize',20,'FontWeight','Bold','interpreter','latex');
set(gca,'Fontsize',16,'FontWeight','Bold','LineWidth',2);


%==========================================================================
% BLACK-SCHOLES PRICE
function BlackScholesPrice=BlackScholesPricer(S0,K,T,r,sigma)

F=S0.*exp(r.*T);
d1=log(F./K)./(sigma.*sqrt(T))+sigma.*sqrt(T)/2;
d2=log(F./K)./(sigma.*sqrt(T))-sigma.*sqrt(T)/2;
BlackScholesPrice = exp(-r.*T).*(F.*normcdf(d1)-K.*normcdf(d2));
%==========================================================================
% BLACK-SCHOLES IMPLIED VOLATILITY OBJECTIVE FUNCTION
function delta = objfcn(volatility, S0, K, T, r, CallPrice)

BSprice = BlackScholesPricer(S0, K, T, r, volatility);
delta = CallPrice - BSprice;
%==========================================================================
