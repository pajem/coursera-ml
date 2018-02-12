function [J, Y] = pauComputeCost(h, y, num_labels)
% computes the cost for a neural network given output activation values h
% and expected values y.
%
% Note: y should be converted to a vetor of 1's and 0's where 1 is the correct
%       classification value

% output
J = 0;

% number of samples
m = size(h, 1);

% convert y to a vector of 1's and 0's
Y = zeros(m,num_labels);

for i = 1:m,
	% set the correct index to 1
	Y(i,y(i)) = 1;
	
	% solve for sample set i
	yp = Y(i,:)';
	hp = h(i,:);

	% accumulate J for all i
	J = J + (log(hp)*(-yp) - log(1 - hp)*(1 - yp));
end;

% function end
end
