function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesis = sigmoid(X*theta);

%theta zero is not regularized
regularization = (lambda/2*m)*sum(theta(2:length(theta)).^2);

J = 1/m*(-1*y'*log(hypothesis)-1*(1-y)'*log(1-hypothesis))+regularization;

%drop theta zero
reg_derived = lambda/m*theta(2:length(theta));

grad_without_reg = 1/m*X'*(hypothesis-y);

%With regularization
grad(1) = grad_without_reg(1);
grad(2:length(grad)) = grad_without_reg(2:length(grad_without_reg))+reg_derived;

% =============================================================

end