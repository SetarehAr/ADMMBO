function [ value ] = Sinosidual( x,y,params )
%UNTITLED16 Summary of this function goes here
%   Detailed explanation goes here

 value=(params(1)*sin(params(2)*pi*(x.^2-params(3).*y))-x-params(4).*y+params(5));
end

