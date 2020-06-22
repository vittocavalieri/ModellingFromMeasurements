function rhs = rhs_syndy(t,x,dummy,xi1,xi2)

A = [x(1); x(2); x(1)*x(2); x(1)^2; x(2)^2; x(1)^2*x(2); x(1)*x(2)^2; x(1)^3; x(2)^3; cos(x(1)); cos(x(2)); sin(x(1)); sin(x(2))];

rhs = [xi1'*A; xi2'*A];

end