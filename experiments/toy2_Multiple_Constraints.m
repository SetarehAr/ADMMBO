
%% Initial setup
clc;
clear all;
currendir  = pwd;
if isempty(strfind(currendir, '/'))     
    filesep = '\';
else
    filesep = '/';
end
idcs= strfind(currendir,filesep);
main_dir= currendir(1:idcs(end)-1);
addpath(genpath(main_dir))


%% Adding required pathes for bayesopt & gpml toolbox

addpath(genpath(strcat(main_dir,filesep,'Direct')));
addpath(genpath(strcat(main_dir,filesep,'FMINSEARCHBND')));
addpath(genpath(strcat(main_dir,filesep,'admmbo')));
addpath(genpath(strcat(main_dir,filesep,'bayesopt')));
addpath(genpath(strcat(main_dir,filesep,'functions')));
addpath(genpath(strcat(main_dir,filesep,'initial-data')));
addpath(genpath(strcat(main_dir,filesep,'gpml')));
addpath(genpath(strcat(main_dir,filesep,'gpml',filesep,'cov')));
addpath(genpath(strcat(main_dir,filesep,'gpml',filesep,'doc')));
addpath(genpath(strcat(main_dir,filesep,'gpml',filesep,'inf')));
addpath(genpath(strcat(main_dir,filesep,'gpml',filesep,'lik')));
addpath(genpath(strcat(main_dir,filesep,'gpml',filesep,'mean')));
addpath(genpath(strcat(main_dir,filesep,'gpml',filesep,'prior')));
addpath(genpath(strcat(main_dir,filesep,'gpml',filesep,'util')));

%% Experimantal setup
rho=.1;
M=20;
initial_bud_f=10;
final_bud_f=2;
initial_bud_c=20;
final_bud_c=2;
InfPoint=[-1000 -1000];
initial_num = 2;
num=5; 
y=0;
tolC = 0.01;

%% Initializing parameters and samples
% load initial data
load('InitialData_toy2.mat')
bounds=[0 1;0 1];
dim = size(bounds,1);

%% Defining a struct called "problem" for setting the objective and constraint functions & bounds
problem.f='Sum_Linear';
problem.f_parameters=[1,1,0];
problem.c={'Sinosidual','Circular'};
problem.c_parameters={[-0.5;2;2;2;1.5],[0;0;1.5]};
problem.bounds=bounds;
problem.InfPoint=(bounds(:,1)'+1000);
total_budget=100*(length(problem.c)+1);
Iters = 1+ round((total_budget - (initial_bud_c*(length(problem.c))) - initial_bud_f)/((length(problem.c)+1) *final_bud_c));


funcf=str2func(problem.f);
F = @(X) funcf(X(1),X(2),problem.f_parameters);
C = cell(length(problem.c),1);
for i=1:length(problem.c)
    func=str2func(problem.c{i});
    C{i}= @(X) func(X(1),X(2),problem.c_parameters{i});
    clear func
end

problem.F = F;
problem.C = C;
%% Defining a struct called "opt" for setting the optmization parameters 
%%%%%%%%%%%%%%%%%%%%%%%%
% Objective struct opt.f
%%%%%%%%%%%%%%%%%%%%%%%%
opt.f.grid_size = 10^5; % Size of grid to select candidate hyperparameters from.
opt.f.max_iters =[]; % Maximum number of function evaluations.
opt.f.step_iters=initial_bud_f;
opt.f.reduced_step_iters=final_bud_f;% 
opt.f.meanfunc = {@meanShiftedBar}; % Constant mean function.
opt.f.covfunc = {@covMaterniso, 5};
opt.f.hyp = -1; % Set hyperparameters using MLE.
opt.f.dims = dim; % Number of parameters.
opt.f.mins = problem.bounds(:,1)'; % Minimum value for each of the parameters. Should be 1-by-opt.dims
opt.f.maxes = problem.bounds(:,2)'; 
opt.f.save_trace = false;
%opt.f.optimize_ei = true; 
for j=1:length(problem.c)
    opt.f.y{j}=ones(1,dim).*y;
    opt.f.z{j}=bounds(:,1)';
end
    opt.f.rho=rho;
%%%%%%%%%%%%%%%%%%%%%%%%    
% Constraint struct opt.c 
%%%%%%%%%%%%%%%%%%%%%%%%%%
   for j =1:length(problem.c)
    opt.c{j}.grid_size = 10^3;
    opt.c{j}.max_iters =[]; % Maximum number of function evaluations.
    opt.c{j}.step_iters=initial_bud_c;
    opt.c{j}.reduced_step_iters=final_bud_c;% 
    opt.c{j}.meanfunc = {@meanZero}; % Constant mean function.
    opt.c{j}.covfunc ={@covMaterniso, 5};
    opt.c{j}.dims=2;
    opt.c{j}.optimize_ei = false;
     
   end
%%%%%%%%%%%%%%%%%%%%%%
% ADMM struct opt.ADMM
%%%%%%%%%%%%%%%%%%%%%% 
opt.ADMM.max_iters=Iters;% Maximum number of ADMM iterations
opt.ADMM.rho=rho;% ADMM parameter
opt.ADMM.M=M; % ADMM parameter
    
%% Running ADMMBO Algorithm
Samples=cell(num,1);
best_point=cell(num,1);
best_F=cell(num,1);

% Main ADMMBO loop
for i=1:num
    fprintf('#######################################\n')
    fprintf('START OF random run %d:\n',i);
    fprintf('#######################################\n')

    % Selecting the initial points from X0
    initial_samples =X0((i*initial_num)-1:(i*initial_num),:);
    opt.f.initial_points=initial_samples;
    for j =1:length(problem.c)
        opt.c{j}.initial_points=initial_samples;
        %for k =1:initial_num
        %    opt.c{j}.values(k,:)= C{j}(initial_samples(k,:));
        %end
    end

    % Run ADMMBO
    [Samples{i},best_point{i},best_F{i}] = ADMMBO(problem,opt);
    
    % Report the results
    feasible_sum = 0;
    for k =1:length(problem.c)
        feasible_sum = feasible_sum + (C{k}(best_point{i}) <= tolC);      
    end
    
    feaibility_check = (feasible_sum == length(problem.c));
    fprintf('##############################################################################\n')
    fprintf('Best objective found by ADMMBO in run number %d is %.2f',i,best_F{i});
    fmt = [', corresponding to point[', repmat('%.2g, ', 1, numel(best_point{i})-1), '%.2g]\n'];
    fprintf(fmt, best_point{i});
    if feaibility_check ==1
        fprintf('which is feasible.\n');
    else
        fprintf('which is infeasible.\n');
    end 
    fprintf(' %d Samples have been tried.\n',length(Samples{i}));
    fprintf('##############################################################################\n')
end

%% Reporting & Saving the results
fprintf('##############################################################################\n')
fprintf('                    ******************************                    \n')
fprintf('The average optimal objective value among %d random initializations found by ADMMBO is %.2f\n',num, mean(cell2mat(best_F)));
fprintf('                    ******************************                    \n')
fprintf('##############################################################################\n')


FileName='toy2_results_ADMMBO.mat';
save_dir = strcat(main_dir,filesep,'results');
save(fullfile(save_dir,FileName),'Samples','best_F','best_point','problem','opt','F','C','num');

%% To close matlab and clear all added paths
!matlab &
exit
