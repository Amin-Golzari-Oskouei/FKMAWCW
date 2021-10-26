%% Algorithm parameters.

% Algorithm parameters.
% for get best, rsult we use parameters as fallows:
%==========================================================================
% outputs
% phi: the inverse standard deviation from the mode (SDM).        
% Beta: Beta is the distance function coefficient (0.5<=beta<=1).           
% q: the value for the feature weight updates.         
% k: number of clusters.
%==========================================================================

function [phi, Beta, q, k, landa, PX] = Algorithm_parameters(name, class, X, d, N)
 
if strcmp(name , 'balance.mat')
    phi = 1; Beta = 0.9; q = 4;
elseif strcmp(name , 'Car_evaluation.mat')
    phi = 1; Beta = 0.9; q = -4;
elseif strcmp(name , 'chess.mat')
    phi = 0.1; Beta = 0.99; q = -8;
elseif strcmp(name , 'dermatology.mat')
    phi = 0.0001; Beta = 0.99; q = -10;
elseif strcmp(name , 'lungcancer.mat')
    phi = 1; Beta =  1; q = -2;
elseif strcmp(name , 'lymphography.mat')
    phi = 0.001; Beta = 0.99; q = -6;
elseif strcmp(name , 'mushroom.mat')
    phi = 0.01; Beta = 0.9; q = -8;
elseif strcmp(name , 'nursery.mat')
    phi = 0.01; Beta = 0.9; q = 4;
elseif strcmp(name , 'soybean.mat')
    phi = 0.1; Beta = 0.99; q = -8;
elseif strcmp(name , 'vote.mat')
    phi = 0.0001; Beta = 0.9; q = -8;
elseif strcmp(name , 'zoo.mat')
    phi = 0.01; Beta = 0.9; q = 2;
end

for i=1:d
    freq = hist(X(:,i),unique(X(:,i)));
    SDM(1,i) = 1 - ((1./(N.^2 .* (nnz(freq)-1))) .* sum((max(freq)-freq).^2)).^0.5;
end
landa = phi ./ SDM;

%probablity of px
for i=1:N
    PX(i,:)=sum(X==repmat(X(i,:),N,1),1);
end

k=size(unique(class),1);        % number of clusters.

end
