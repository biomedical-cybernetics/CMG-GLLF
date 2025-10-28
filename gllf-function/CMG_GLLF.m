function y = CMG_GLLF(x, mu, xmin, xmax, yL, yR, I, rectify)

% Cannistraci-Muscoloni-Gu model for the generalized logistic-logit function

% %% INPUT %%%
% x - numerical vector of the values at which the function is evaluated, xmin<=x<=xmax.

% mu - type-growth-rate parameter, 0<=mu<=1.
%   mu=0:    step function 
%   0<mu<0.5:  generalized logistic function
%   mu=0.5:    linear function
%   0.5<mu<1:    generalized logit function
%   mu=1:  constant function 

% xmin - minimum x value allowed.
% xmax - maximum x value allowed.
% yL - value of the function at xmin.
% yR - value of the function at xmax.

% I - inflection parameter, 0<=I<=1 (default: I=0.5).
%   it represents the proportion of the x-axis range [xmin,xmax] at which the inflection occurs.
%   in particular, the inflection point of the logistic-logit function is at coordinates:
%   xI = xmin+I*(xmax-xmin)
%   yI = yL+I*(yR-yL)

% rectify - 1 or 0 to indicate if the function should be rectified or not (default: rectify=0).

% %% OUTPUT %%%
% y - numerical vector of the values of the function evaluated at x.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check input
narginchk(6,8);
validateattributes(x,{'numeric'},{'ndims', 3});
validateattributes(mu,{'numeric'},{'scalar','>=',0,'<=',1});
validateattributes(xmin,{'numeric'},{'scalar'});
validateattributes(xmax,{'numeric'},{'scalar'});
if xmax<=xmin; error('xmax must be greater than xmin.'); end
if any(x(:)<xmin) || any(x(:)>xmax); error('x values must be between xmin and xmax.'); end
validateattributes(yL,{'numeric'},{'scalar'});
validateattributes(yR,{'numeric'},{'scalar'});
if ~exist('I','var') || isempty(I); I = 0.5;
else; validateattributes(I,{'numeric'},{'scalar','>=',0,'<=',1}); end
if ~exist('rectify','var') || isempty(rectify); rectify = 0;
else; validateattributes(rectify,{'logical','numeric'},{'scalar','binary'}); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if mu == 1
    % constant function
    y = repmat(yL+I*(yR-yL),size(x));
elseif mu == 0
    % step function
    y = zeros(size(x));
    xI = xmin+I*(xmax-xmin);
    y(x<xI) = yL;
    y(x>xI) = yR;
    y(x==xI) = max(yL,yR);
elseif mu <= 0.5
    % generalized logistic function (linear function for mu=0.5)
    y = yL + (yR-yL) ./ (1 + (xmax-x)./(x-xmin).*exp((2-1./mu).*((x-xmin)./(xmax-xmin)-I)));
else
    % generalized logit function (computational approximation)
    if yL < yR
        ys = linspace(yL,yR,1000);
        xs = xmin + (xmax-xmin) ./ (1 + (yR-ys)./(ys-yL).*exp((2-1./(1-mu)).*((ys-yL)./(yR-yL)-I)));
    else
        ys = linspace(yR,yL,1000);
        xs = xmin + (xmax-xmin) ./ (1 + (yL-ys)./(ys-yR).*exp((2-1./(1-mu)).*((ys-yR)./(yL-yR)-I)));
        ys = yR + yL - ys;
    end
    
    % for mu -> 1, there are cases when multiple y values collapse on the same x value
    % if (yL<yR & x<xI) | (yL>yR & x>=xI), take the lowest of these y values
    % if (yL<yR & x>=xI) | (yL>yR & x<xI), take the highest of these y values
    % this choice ensures to reach the extreme points [xmin,yL] and [xmax,yR]
    y = NaN(size(x));
    xI = xmin+I*(xmax-xmin);
    mask = x<xI;
    [~,idx] = min(abs(reshape(x(mask),1,[])-reshape(xs,[],1)),[],1);
    y(mask) = ys(idx);
    mask = x>=xI;
    [~,idx] = min(abs(reshape(x(mask),1,[])-reshape(xs(end:-1:1),[],1)),[],1);
    ys = ys(end:-1:1);
    y(mask) = ys(idx);
end

if rectify
    y = max(y,0);
end
