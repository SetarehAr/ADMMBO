function [ value ] = C_4d( x,y,z,u,params )
%UNTITLED16 Summary of this function goes here
%   Detailed explanation goes here
%value=3+20*exp(-0.2*sqrt(.25*((3*x-1)^2+(3*y-1)^2+(3*z-1)^2+(3*u-1)^2)))...
 %   +exp(.25*((cos(2*pi*(3*x-1)))+(cos(2*pi*(3*y-1)))+(cos(2*pi*(3*z-1)))+(cos(2*pi*(3*u-1)))))-20-exp(1);
C=[1;2;3;3.2];
a=[10 .05 3 17;3 10 3.5 8;17 17 1.7 .05;3.5 .1 10 10];
p=[.131 .232 .234 .404;.169 .413 .145 .882;.556 .830 .352 .873;.012 .373 .288 .574];

for i=1:4
    c=C(i);
    if i==1
        t=x;
    elseif i==2
        t=y;
    elseif i==3
        t=z;
    else
        t=u;
    end
        
    for j=1:4
      temp(j)=a(j,i)*(t-p(j,i))^2;  
    end
 valC(i)=c*exp(-sum(temp));
 
 clear temp c
end
value=-1*((1/0.8387)*(-1.1+sum(valC)));
% value=(params(1)*sin(params(2)*pi*(x.^2-params(3).*y))-x-params(4).*y+params(5));
end

