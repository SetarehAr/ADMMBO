function [Samples,best_point,best_F] = ADMMBO(problem,opt)  
%% runs the ADMMBO algaorithm given initial points & budgets using EI

    fprintf('#######################################\n')
    fprintf('START OF "ADMMBO" ALGORTIHM\n');
    %% Preallocation
    ABSTOL=0.05; % ADMM Convergence In Practice Parameter
    RELTOL=0.05; %ADMM Convergence In Practice Parameter
    relaxp=1.5; % if >1 sets a faster convergence
    relaxd=1.5;
    C_check=ones(length(problem.c),1);
    con=zeros(length(problem.c),1);
    mu=10; % Adapting rho Parameter
    thai=2; % Adapting rho Parameter
    Samples=[];
    tolC = 0.01;

    %% rebuildig the functions & evaluate the function values for an infeasible point
    F = problem.F;
    C=problem.C;
    best_point = problem.InfPoint;
    best_F = 1000 * F(best_point) * sign(F(best_point)); % a very positive value
    best_C = ones(length(problem.c),1);
    for s=1:length(problem.c)
        best_C(s) = C{s}(best_point)*1000 * sign(C{s}(best_point));
    end  

    %% ADMM Main Loop
     for i=1:opt.ADMM.max_iters
        fprintf('#######################################\n')
        fprintf('Solving the Optimality subproblem at outer iteration %d of ADMMBO:\n',i);
        fprintf('    ******************************    \n')
        fprintf('Bayesian Optimizations inner iterations starts:\n')
        fprintf('    ******************************   \n')
        
        %% X-update 
        % building the Augmented Lagrangian function
        ybar=mean(cat(3,opt.f.y{:}),3);
        zbar=mean(cat(3,opt.f.z{:}),3);
        AL_X = @(X) F(X)+(length(problem.c)*opt.ADMM.rho/2)*norm(X-(zbar)+(ybar/opt.ADMM.rho))^2;
 
        if i==1
            opt.f.max_iters=size(opt.f.initial_points,1)+(opt.f.step_iters);
        else
            opt.f.max_iters=size(opt.f.initial_points,1)+(opt.f.reduced_step_iters);
        end

         % Optimizing the AUGMENTED LAGRANGIAN wrt X using BO
        [xmin,~,T_u]=bayesopt(AL_X,opt.f); 
        XX(i,:)=xmin;

        % Updating the Samples set based on X-step BO results
        oldSamplesNum=size(T_u.samples,1);
        if i==1    
            Samples=[Samples;T_u.samples];
        else
            Samples=[Samples;T_u.samples(oldSamplesNum+1:end,:)];
        end
        
        % updating x*_i for Z-update and Y-updates & gathering the
        % observed data
        opt.f.x=xmin;% Updating x* for Z-update and Y-updates
        opt.f.initial_points=T_u.samples;% Updating initial X points

        if i==1
            iter_num = sprintf('%dst',i);
        else
            iter_num = sprintf('%dnd',i);
        end    
          
        % Updating the incumbent 
        [best_point,best_F,best_C] = incumbent_update(best_point,best_F,best_C,xmin,F,C,tolC,iter_num,'Optimality');
        
        %% Z-update 
        fprintf('#######################################\n')
        fprintf('Solving the Feasibility subproblem at %s outer iteration of ADMMBO:\n',iter_num);
        fprintf('    ******************************    \n')
        fprintf('Bayesian Optimizations inner iterations starts:\n')
        fprintf('    ******************************   \n')
        
        % Checking if we have already satisfied the constraint's coonvergence criterion by C_check
        if C_check
            % Keeping track of old Z* to check (Z*(k+1)-Z*(k))
            zold=opt.f.z;
            
            for j=1:length(problem.c)
                
                % Adapting the max number of BO iterations according to ADMMBO's inner loop
                if i==1
                    opt.c{j}.max_iters=opt.c{j}.step_iters;
                else
                    opt.c{j}.max_iters=opt.c{j}.reduced_step_iters;  
                end
                
                % Optimizing the feasibility subproblem j^th  wrt Z using BO
                [zmin{j},~,T_h{j}] = bayesfeas(problem,opt,j); 
                
                if j ==1
                    subproblem = [ sprintf('%dst',j) ' Feasibility'];
                else
                    subproblem = [ sprintf('%dnd',j) ' Feasibility'];
                end
                
                
                if i==1    
                    Samples=[Samples;T_h{j}.samples];
                else
                    Samples=[Samples;T_h{j}.samples(oldSamplesNumC{j}+1:end,:)];
                end
                
                ZZ{j}(i,:)=opt.f.z{j};
                % Updating the Samples set based on Z-step BO results
                oldSamplesNumC{j}=size(T_h{j}.samples,1);
                opt.f.z{j}=zmin{j};% Updating z* for X-update and Y-updates
                opt.c{j}.initial_points=T_h{j}.samples;% Updating initial Z points
                
                % Updating the incumbent 
                [best_point,best_F,best_C] = incumbent_update(best_point,best_F,best_C,zmin{j},F,C,tolC,iter_num,subproblem);
                
                %% Y-update 
                ymin = opt.f.y{j} +opt.ADMM.rho*(xmin - zmin{j});
                opt.f.y{j}=ymin;
                clear  ymin
            end
        end
        %% Check the termination Condition
        for j=1:length(problem.c) 
             history.r_norm{j}= norm(opt.f.x-opt.f.z{j});
             history.s_norm{j}=norm(-opt.ADMM.rho*(opt.f.z{j}- zold{j}));
             history.eps_pri{j} = sqrt(opt.f.dims)*ABSTOL +relaxp*RELTOL*max(norm(opt.f.x), norm(-opt.f.z{j}));
             history.eps_dual{j}= sqrt(opt.f.dims)*ABSTOL + relaxd*RELTOL*norm(opt.ADMM.rho*opt.f.y{j});
             if (history.r_norm{j} < history.eps_pri{j} && ...
              history.s_norm{j} < history.eps_dual{j})
              con(j)=1;
             end
        end
         
      % checking the ADMM convergence
     if con
       fprintf("It takes %d ADMM iterations To converge.\n ",i);
         break;
     end
     
     % updating the penalty parameter
     if mean(cell2mat(history.r_norm)) > mu*mean(cell2mat(history.s_norm))
        opt.ADMM.rho= opt.ADMM.rho*thai;
        fprintf("rho is increased\n");
     elseif  mu* mean(cell2mat(history.r_norm)) < mean(cell2mat(history.s_norm))
        opt.ADMM.rho= opt.ADMM.rho/thai;
        fprintf("rho is decreased\n");
     else
        opt.ADMM.rho= opt.ADMM.rho;
        fprintf("rho is unchanged\n");
     end
     
        
       
     end
end

function[best_point_updated,best_F_updated,best_C_updated] = incumbent_update(best_point,best_F,best_C,current_point,F,C,tolC,iter_num,subproblem)
    % this function updates and reports the incumbent and its objective value
    current_F = F(current_point);
    current_C = ones(length(C),1);
    for s=1:length(C)
    current_C(s) = C{s}(current_point);
    end
    if sum(current_C > tolC) && sum(best_C >tolC)
        best_point_updated = best_point;
        best_F_updated = best_F; 
        best_C_updated = best_C;
        fprintf('    ******************************    \n')
        fprintf('No feasible point is found yet after %s ADMMBO iteration during %s subproblem! \n',iter_num,subproblem);
        fprintf('    ******************************   \n')
        
    elseif  sum(current_C>tolC) == 0 && current_F <= best_F
       best_point_updated = current_point;
       best_F_updated = current_F;
       best_C_updated = current_C;
       fprintf('    ******************************    \n')
       fprintf('The incumbent is updated & the bestfeasible oberved value after %s ADMMBO iteration during %s subproblem is %f.\n',iter_num,subproblem,best_F_updated);
       fprintf('    ******************************   \n')
    else
       best_point_updated = best_point;
       best_F_updated = best_F; 
       best_C_updated = best_C;
       fprintf('    ******************************    \n')
       fprintf('The incumbent is NOT updated at %s ADMMBO iteration during %s subproblem. \n',iter_num,subproblem);
       fprintf('    ******************************   \n')
       
         
    end
    
end
