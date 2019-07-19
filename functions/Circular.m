function [ value ] = Circular(x,y,params)
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here
value=  (x-params(1)).^2+(y-params(2)).^2-params(3);
%value=x-.5;
end

