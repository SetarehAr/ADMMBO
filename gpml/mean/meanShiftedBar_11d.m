function [ A] = meanShiftedBar_11d(hyp,x,i)
%UNTITLED2 Summary of this function goes here
if nargin < 2, A= ['23'];return;end 
y_00=hyp(1:11);
z_00=hyp(12:22);

rho=hyp(23);
num=size(x,1);
z=repmat(z_00,num,1);
y=repmat(y_00,num,1);

mat=x-z+(y/rho);
A=(rho/2)*(sum(abs(mat).^2,2));

end

