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
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%first parameter is not regularized
regularization = (lambda/(2*m))*theta(2:end)'*theta(2:end);

%t0+t1*x1+t2*x2...
hypothesis = X*theta;
error = hypothesis-y;
cost = (1/(2*m))*error'*error;

J= cost+regularization;

%GRADIENT
%gradient of regularization
regularization_derived= (lambda/m)*theta(2:end);

%X is a (m x n) matrix if there are n parameters
gradient_without_regularization = (1/m)*X'*error;

%bias term does not get regularized
grad(1) = gradient_without_regularization(1);
grad(2:end) = gradient_without_regularization(2:end)+regularization_derived;
% =========================================================================

grad = grad(:);

end
