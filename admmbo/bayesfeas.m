function [zmin,h_min,botrace] =bayesfeas(problem,opt,const_num)
%% solves one feasiblity problem at each ADMMBO iteration

    samples= opt.c{const_num}.initial_points;
    C_values=zeros(size(samples,1),1);
    h_values=zeros(size(samples,1),1);
    
    C = problem.C;
    h_func=@(Z) (C{const_num}(Z)>0)+(opt.ADMM.rho/(2*opt.ADMM.M))*norm(opt.f.x-Z+(opt.f.y{const_num}/opt.ADMM.rho))^2;
    CNST_func=@(Z) C{const_num}(Z);
    
    
    
    for i=1:size(samples,1)
        fprintf('Running initial point #%d...\n',i);
        h_values(i)=h_func(samples(i,:));
        C_values(i)=CNST_func(samples(i,:));
    end
      
    h_plus=min(h_values);
    
    meanfunc=opt.c{const_num}.meanfunc;
    covfunc = opt.c{const_num}.covfunc; 
    likfunc = @likGauss; 
    hyp.lik = log(.1);   
    ell = 1/4; sf = 1; hyp.cov = log([ell; sf]); 
    
    for i=1:opt.c{const_num}.max_iters
        
        grid_data=zeros(opt.c{const_num}.grid_size,opt.c{const_num}.dims);
       
        for j=1:opt.c{const_num}.dims
            grid_data (:,j)= (problem.bounds(j,2)-problem.bounds(j,1)) .*rand(1,opt.c{const_num}.grid_size,1) + problem.bounds(j,1);
            clear grid_data_aug
        end
        
        hyp2 = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, samples,C_values);
        [m,s2] = gp(hyp2, @infExact, meanfunc, covfunc, likfunc, samples,C_values,grid_data);
    
        EI =EI_Z(m,s2,grid_data,h_plus,opt,const_num);
        
        if opt.c{const_num}.optimize_ei == true
            max_num=10;% number of max EI values to optimize from
            sortedEIValues = unique(sort(EI,'ascend')); % Unique EI values from small to large

            if length(sortedEIValues)>max_num
                maxEIValues = sortedEIValues(end-max_num+1:end);  % Get the max_num largest values
            else
                maxEIValues = sortedEIValues;
            end

            maxIndexbinary = ismember(EI,maxEIValues);  
            maxIndex=find(maxIndexbinary==1); % index of max_num max values in the grid data

            gpMeanVar= @(z) gp(hyp2, @infExact, meanfunc, covfunc, likfunc, samples,C_values,z); % Posterior GP dist for a sample point z based on observed samples Z, C(Z)
            meanz = @(z) cell(GPfinal(z,gpMeanVar)){1} ; % posterior mean of the mentioned distribution
            s2z = @(z) cell(GPfinal(z,gpMeanVar)){2}; % posterior variance of the mentioned distribution

            Qz=@(Z) h_plus-(opt.ADMM.rho/(2*opt.ADMM.M))*norm(opt.f.x-Z+(opt.f.y{const_num}/opt.ADMM.rho))^2;

            Zstars=zeros(max_num,opt.c{const_num}.dims);% Preallocating the optimal points after optimizing EI from top grid data choices
            FinalMinEI_multi=zeros(max_num,1); %  Preallocating the optimal EI values 

            for j=1:max_num
                grid_sample=grid_data(maxIndex(j),:); % each of the samples in the grid data z with a top max_num value of EI(z)
                Qz_val=Qz(grid_sample); % evaluating Qz for that sample

                % Building the EI function based on the value of Qz for that sample
                if Qz_val < 0 
                    EIz_func=@(z) sum(0.*z);
                elseif Qz_val>0 && Qz_val<=1 
                    EIz_func=@(z) -1*(Qz(z).* normcdf(0,meanz(z),s2z(z))); % -1* changes the maximization of the EI to minimization to use fminsearchcbnd
                else
                    EIz_func=@(z) -1*(Qz(z)-(1-normcdf(0,meanz(z),s2z(z))));
                end

                % Minimizing the -1*EI(.) (instead of maximizing EI(.)) starting from the sample of interest
                % Zstar is the optimum for minimized -1*EI    


                    options = optimset('MaxFunEvals',10^9);
                    Zstars(j,:) = fminsearchbnd(EIz_func,grid_sample,problem.bounds(:,1)',problem.bounds(:,2)',options);% updated on Jan 12 for any dims instead of 2
                    FinalMinEI_multi(j)=EIz_func(Zstars(j,:)); % Optimal found EI value

                clear grid_sample Qz_val EIz_func
            end
            [FinalMinEI, FinalMinEI_ind]=min(FinalMinEI_multi); %finding the index of minimum of optimized EI values
            z_opt=Zstars(FinalMinEI_ind,:); 
            EI_val = max(0,-1*FinalMinEI); % for reporting purposes
        else
            [FinalMaxEI, FinalMaxEI_ind]=max(EI);
            z_opt = grid_data(FinalMaxEI_ind,:);
            EI_val = FinalMaxEI;
        end
        
 
      
       %[FinalMinEI, FinalMinEI_ind]=min(FinalMinEI_multi); %finding the index of minimum of optimized EI values

       %z_opt=Zstars(FinalMinEI_ind,:); % Finiding the optimum of the minimzed EI value over all 

       h_opt=h_func(z_opt);
       C_opt=CNST_func(z_opt);
    
       samples=[samples;z_opt]; % we augment the samples with our new proposed point
       h_values=[h_values;h_opt];% we then update the h(samples)
       C_values=[ C_values; C_opt];% we then update the C(samples)
    
      % updating best value observed so far if applicable
        if h_opt<h_plus
           h_plus=h_opt;
        end
        fprintf('Subproblem %d, Iteration %d, ei = %f, value = %f, overall min = %f\n',const_num,i+size(opt.c{const_num}.initial_points,1),EI_val,h_opt,h_plus);

        clear FinalMinEI_ind FinalMinEI_multi z_opt EI_max m s2 
        clear grid_data_x grid_data_y grid_data m s2 hpy2 EI  sortedEIValues maxEIValues maxIndexbinary maxIndex grid_sample Qz_val EIz_func
        clear  gpMeanVar meanz s2z Qz  Zstars FinalMinEI_multi FinalMinEI_ind z_opt Zstars h_opt C_opt F_opt EI_val
    end
 
    [h_min,min_Ind]=min(h_values);
    zmin=samples(min_Ind,:);
    botrace.samples = samples;
    botrace.values = C_values;

end

  function F = GPfinal(x,gpMeanVar)
     [mean_gp,var_gp]=gpMeanVar(x);
        F = {mean_gp, var_gp};
  end



