function [ A] = meanShifted7d(hyp,x,i)
%UNTITLED2 Summary of this function goes here
if nargin < 2, A= ['15'];return;end 
y_00=hyp(1:7);
z_00=hyp(8:14);
rho=hyp(15);
num=size(x,1);
z=repmat(z_00,num,1);
y=repmat(y_00,num,1);

mat=x-z+(y/rho);
A=(rho/2)*(sum(abs(mat).^2,2));

end

