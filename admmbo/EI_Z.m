function [ EI] = EI_Z(mean_grid,variance_grid,grid_data,h_plus,opt,const_num)
%% The goal of this function is to evaluate EI(.) for the grid data

    % preprocessing 
    Qz=@(Z) h_plus-(opt.ADMM.rho/(2*opt.ADMM.M))*norm(opt.f.x-Z+(opt.f.y{const_num}/opt.ADMM.rho))^2;
    Qz_val=zeros(opt.c{const_num}.grid_size,1);
    
    for j=1:opt.c{const_num}.grid_size
        Qz_val(j,1)=Qz(grid_data(j,:));
    end
  
    % finding the EI() regions
    ind_2=find(Qz_val>=0 & Qz_val<1);
    ind_3=find(Qz_val>=1);
    
    % Evaluating EI(z) based on regions
    cdf_at_0=normcdf(0,mean_grid,variance_grid);
    EI=zeros(length(Qz_val),1);
    EI(ind_2)=Qz_val(ind_2).*cdf_at_0(ind_2); 
    EI(ind_3)=Qz_val(ind_3)-(1-cdf_at_0(ind_3));
    
end

