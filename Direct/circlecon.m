%------------------------------------------------------%
% DIRECT Sample Call circlecon.m                       %
% Purpose: Return the value of the function            %
% c(x) = x^2 + y^2 - 1                                 %
%                                                      %
% You will need the accompanying programs              %
% Direct.m and gp.m to run this code                   %
% successfully.                                        %
%                                                      %
% These codes can be found at                          %
% http://www4.ncsu.edu/~definkel/research/index.html   %
%------------------------------------------------------%
function retval = circlecon(x)

%retval = (x(1)-2)^2 + (x(2)-2)^2 - 4;
%retval = (-0.5*sin(2*pi*(x(1)^2-2*x(2)))-x(1)-2*x(2)+2);
retval=sin(x(1)).*sin(x(2))+.95;