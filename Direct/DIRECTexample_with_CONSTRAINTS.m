%------------------------------------------------------%
% DIRECT Sample Call                                   %
% Purpose: Find the Global Minimum value of            %
%          the Goldstein-Price Test Function           %
%          using the DIRECT algorithm                  %
%          subject to the constraint                   %
%          x^2 + y^2 - 1 <= 0                          %
%          USES L1 PENALTY FUNCTIONS                   %
%                                                      %
% You will need the accompanying programs              %
% Direct.m, circlecon.m and gp.m to run this code      %
% successfully.                                        %
%                                                      %
% These codes can be found at                          %
% http://www4.ncsu.edu/~definkel/research/index.html   %
%------------------------------------------------------%
clear all;
% 1. Establish bounds for variables
bounds = [0 6;0 6];

% 2. Send options to Direct
%    We tell DIRECT that the globalmin = 3
%    It will stop within 0.01% of solution
%options.testflag  = 1;
%options.globalmin = 3;
%options.showits   = 1;
%options.tol       = 0.01;

% 2a. NEW!
% Pass Function as part of a Matlab Structure
Problem.f = 'gp';
Problem.numconstraints = 1;
Problem.constraint(1).func    = 'circlecon'; %constraint function
Problem.constraint(1).penalty = 1;           %penalty value you choose

%-------------------------------------------------------%
% If there was another constraint,                      %
% it would be passed like this:                         %
%                                                       %
% Problem.constraint(2).func = 'anotherconstraint';     %
% Problem.constraint(2).penalty = 1;                    %
%                                                       %
% Also, Problem.numconstraints would need               %
% to be updated                                         %
%-------------------------------------------------------%

% 3. Call DIRECT
%[fmin,xmin,hist] = Direct(Problem,bounds,options);
[fmin,xmin,hist] = Direct(Problem,bounds);

% 4. Plot iteration statistics
plot(hist(:,2),hist(:,3))
xlabel('Fcn Evals');
ylabel('f_{min}');
title('Iteration Statistics for GP test Function');