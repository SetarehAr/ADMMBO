function [ value ] = sinXsinY(x,y,params)
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here
%value=  (x-params(1)).^2+(y-params(2)).^2-params(3);
value=sin(x).*sin(y)+params;
%value=x-.5;
end

