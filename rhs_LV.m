function rhs = rhs_LV(t,x,dummy,b,p,r,d)

rhs = [(b-p*x(2))*x(1); (r*x(1)-d)*x(2)];

end