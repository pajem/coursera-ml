function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% compute for h
[a2, z2] = pauComputeLayer(X,Theta1);
[a3, z3] = pauComputeLayer(a2, Theta2);
h = a3; 

% compute cost
[J, Y] = pauComputeCost(h, y, num_labels);

% regularize
t1 = pauSquaredTheta(Theta1);
t2 = pauSquaredTheta(Theta2);

% regularize cost
J = J/m + (t1 + t2)*(lambda/(2*m));

% backpropagation
x_bias = [ones(m,1), X];
a2_bias = [ones(m,1), a2];


Theta1_nobias = Theta1(:,2:size(Theta1,2));
Theta2_nobias = Theta2(:,2:size(Theta2,2));
D1 = Theta1_grad;
D2 = Theta2_grad;

X_bias = [ones(m,1), X];
i = 0;
for i = 1:m,
	% compute delta3
	a3_i = a3(i,:);
	Y_i = Y(i,:);
	d3_i = a3_i - Y_i;

	% compute delta2
	a2_i = a2(i,:);
	z2_i = z2(i,:);
	zg2_i = sigmoidGradient(z2_i);
	d2_i = (d3_i * Theta2_nobias) .* zg2_i;

	% accumulate D2
	a2_bias_i = a2_bias(i,:);
	D2 = D2 + d3_i' * a2_bias_i;

	% accumulate D1
	x_i = X_bias(i,:);
	D1 = D1 + d2_i' * x_i;
end;

th1 = Theta1;
th2 = Theta2;
th1(:,1) = 0;
th2(:,1) = 0;

D1 = D1 ./ m + (th1) .* (lambda/m);
D2 = D2 ./ m + (th2) .* (lambda/m);

Theta1_grad = D1;
Theta2_grad = D2;

%{
delta3 = h - y;
zg2 = sigmoidGradient(z2);
t2_nobias = Theta2(:,2:size(Theta2,2));

delta2 = (delta3 * t2_nobias) .* zg2;

a1 = X_withbias;
th1 = Theta1;
th2 = Theta2;
th1(:,1) = 0;
th2(:,1) = 0;
Theta1_grad = (delta2' * a1) ./ m + (lambda/m) .* (th1);
Theta2_grad = (delta3' * a2) ./ m + (lambda/m) .* (th2);
%}
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
