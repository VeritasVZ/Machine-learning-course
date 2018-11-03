function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

trans_X_X = X'*X % X'= (n x m ),  X = (m x n), res = n x n

theta = inv(trans_X_X)*X'*y  %nxn * nxm * mx1  = nx1

% -------------------------------------------------------------


% ============================================================

end
