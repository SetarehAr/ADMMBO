function [ A] = meanShiftedBar(hyp,x,i)
%UNTITLED2 Summary of this function goes here
%tmp = num2str(length(hyp.mean));
if nargin < 2, A= ['5'];return;end 

y_00=hyp(1:size(x,2));
z_00=hyp((size(x,2)+1):(2*size(x,2)));
rho=hyp(end);
num=size(x,1);
z=repmat(z_00,num,1);
y=repmat(y_00,num,1);

mat=x-z+(y/rho);
A=(rho/2)*(sum(abs(mat).^2,2));

end

