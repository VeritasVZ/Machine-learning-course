function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h_arg = X*theta;
h = sigmoid(h_arg);
error1 = (-1*y)'*log(h);
error2 = (1-y)'*log(1-h);
unreg_J = 1/m*(sum(error1)-sum(error2));
theta(1) = 0;
theta_sqt = theta'*theta;
%reg_term = (2*m)*sum(theta_sqt);
%J = unreg_J + reg_term;
J = unreg_J;

error = h-y;
%grad =  (1/m)*(X'*error) + (1/m)*theta;
grad = (1/m)*(X'*error);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
