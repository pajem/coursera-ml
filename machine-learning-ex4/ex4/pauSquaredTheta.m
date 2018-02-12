function t = pauSquaredTheta(theta)
% computes them summation of squared thetas for layer l.
% For use in regularization

% remove bias unit
theta = theta(:,2:size(theta,2));

% square
theta = theta .^ 2;

% summation
theta = theta * ones(size(theta,2),1);
t = sum(theta,1);

% function end
end
