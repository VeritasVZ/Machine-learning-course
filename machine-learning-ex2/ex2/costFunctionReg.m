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
h_arg = X*theta;
h = sigmoid(h_arg);
error1 = (-1*y)'*log(h);
error2 = (1-y)'*log(1-h);
unreg_J = 1/m*(sum(error1)-sum(error2));
theta(1) = 0;
theta_sqt = theta'*theta;
reg_term = lambda*(2*m)*sum(theta_sqt);
J = unreg_J + reg_term;


error = h-y;
grad =  (1/m)*(X'*error) + (lambda/m)*theta;







% =============================================================

end
