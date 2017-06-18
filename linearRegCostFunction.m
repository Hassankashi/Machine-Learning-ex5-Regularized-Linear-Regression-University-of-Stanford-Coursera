function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
%%%%%%%%
% Remove theta 0 due to it is related to bias vector
% first row with all columns   === [] null
thetaprim=zeros(size(theta));
thetaprim= theta;
thetaprim(1,:)= [];
stheta=sum(thetaprim.^2);
costT=(lambda/(2*m))*stheta;
J=(1/(2*m))*sum((X*theta - y).^2)+costT;

grad = zeros(size(theta));

gardaxi=((X*theta - y)'*X);

if J==0
    %grad(1,1) = gardaxi/m ;
    
  grad(1,1) = sum((theta'*X'-y').*X(:,1)')/m;
  
 grad(2:end,1)= (((theta'*X'-y')*X(:,2:end))/m);     
    
else 
    grad(1,1) = sum((theta'*X'-y').*X(:,1)')/m;
  
 grad(2:end,1)= (((theta'*X'-y')*X(:,2:end))/m)+((lambda/m)*theta(2:end,1))';                    

end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
