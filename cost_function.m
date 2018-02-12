function j = cost_function(theta,x,y)
	m = length(y);
	j = sum((x*theta-y).^2)/(2*m)
endfunction
