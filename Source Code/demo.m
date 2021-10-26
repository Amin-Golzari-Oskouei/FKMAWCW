% This demo implements the CGFFCM algorithm as described in
% A.Golzari oskouei, M.Balafar, and C.Motamed, "FKMAWCW: Categorical Fuzzy k-Modes Clustering with Automated 
% Attribute-weight and Cluster-weight Learning", Chaos, Solitons and Fractals,
% 2021.
%
% Courtesy of A.Golzari

clc
clear all
close all

%% Get list of all mat files in this directory
% DIR returns as a structure array.  You will need to use () and . to get
% the file names.
datafiles = dir('*.mat');
nfiles = length(datafiles);    % Number of files found

%% General Setting
p_init = 0;                     % initial p.
p_max = 0.5;                    % maximum p.
p_step = 0.01;                  % p step.
t_max = 100;                    % maximum number of iterations.
Restarts = 10;                   % number of CGFFCM restarts (default = 1).
fuzzy_degree = 2;               % fuzzy membership degree


for ii=1:nfiles 
    %% Load datasets.
    currentclass = strcat(datafiles(ii).folder,'\',datafiles(ii).name);
    X=load(currentclass);
    X=struct2cell(X);
    X = X{1};
    class=X(:,end);
    [~, ~, ic] = unique(class);
    class = (reshape(ic,[1,size(X,1)]))';
    X(:,end)=[];
    [N,d]=size(X);
    [~, ~, ic] = unique(X);
    X = reshape(ic,[N,d]);
    
    %% Specific parameters for each data set
    [phi, Beta, q, k, landa, PX] = Algorithm_parameters(datafiles(ii).name, class, X,d,N);
    
    %% Cluster the instances using the FWCWFKM procedure.
    for repeat=1:Restarts
        fprintf('==========================CGFFCM==============================\n')
        fprintf('Datset %s: Restart %d\n',datafiles(ii).name(1:end-4), repeat);
        fprintf('...\n')
        
        %Randomly initialize the cluster centers.
        rand('state',repeat)
        tmp=randperm(N);
        M=X(tmp(1:k),:);
        
        %Execute FWCWFKM.
        %Get the cluster assignments, the cluster centers and the cluster variances.
        
        start_CGFFCM = tic; %Timer(start)
        
        [Cluster_elem,M,EW_history,W,Z]= FWCWFKM(X,M,k,p_init,p_max,p_step,t_max,N,fuzzy_degree,d,q,PX,landa,Beta);
        
        time_ave(repeat) = toc(start_CGFFCM); %Timer (end)
        
        [~,Cluster]=max(Cluster_elem,[],1);  %Final clusters.
        
        %Meaures
        EVAL = Evaluate(class,Cluster');
        accuracy(repeat)=EVAL(1);
        ri_adjusted(repeat)=EVAL(2);
        precision(repeat)=EVAL(3);
        recall(repeat)=EVAL(4);
        
        fprintf('End of Restart %d\n',repeat);
        fprintf('========================================================\n\n')
    end
    
    fprintf('Average accurcy score over %d restarts: %f.\n',Restarts,mean(accuracy(accuracy~=inf)));
    fprintf('Average adjusted rand index over %d restarts: %f  .\n',Restarts,mean(ri_adjusted(ri_adjusted~=inf)));
    fprintf('Average precision over %d restarts: %f.\n',Restarts,mean(precision(precision~=inf)));
    fprintf('Average recall over %d restarts: %f.\n',Restarts,mean(recall(recall~=inf)));
    fprintf('Average run time over %d restarts: %f.\n',Restarts,mean(time_ave));
    fprintf('========================================================\n\n')
    
    reslts(ii,:) = {datafiles(ii).name(1:end-4); 
        round(mean(accuracy(accuracy~=inf)),2 );
        round(mean(ri_adjusted(ri_adjusted~=inf)), 2);
        round(mean(precision(precision~=inf)),2);
        round(mean(recall(recall~=inf)),2);
        round(mean(time_ave),2)};
end

%% Save statistical results for all datasets.
% name of saved file is Results.
Final_Results = cell2table(reslts);
Final_Results.Properties.VariableNames(1:6) = {'Dataset', 'Acc', 'ARI ', 'PR', 'RE', 'Run_Time'};
writetable(Final_Results,'Results.txt','Delimiter',' ');

