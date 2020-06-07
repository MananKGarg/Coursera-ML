function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    hx = X*theta - y;
    delta1 = sum(hx.*X(:,1));
    delta2 = sum(hx.*X(:,2));
    delta3 = sum(hx.*X(:,3));
    
    delta1 = delta1/m;
    delta2 = delta2/m;
    delta3 = delta3/m;
    
    delta = [delta1;delta2;delta3];
    
    theta = theta - alpha*delta









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
