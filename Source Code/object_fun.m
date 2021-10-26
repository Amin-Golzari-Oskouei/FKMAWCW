function j_fun = object_fun(N,d,k,Cluster_elem,M,fuzzy_degree,W,Z,q,p,X,PX,landa,Beta)
for j=1:k
    delta(j,:,:)= bitxor(repmat(M(j,:),N,1),X);
    delta(delta(:,:,:)==0) =1-Beta;
    delta(delta(:,:,:)==1) =Beta;
    PM=sum(X==repmat(M(j,:),N,1),1);
    distance(j,:,:)= (1-exp((-1.*repmat(landa,N,1)).*((reshape(delta(j,:,:),[N,d]) .* ( ((PX./N) .* ((PX-1)./(N-1))) + ((repmat(PM,N,1)./N) .* ((repmat(PM,N,1)-1)./(N-1))) )).^2)));
    WBETA = transpose(Z(j,:).^q);
    WBETA(WBETA==inf)=0;
    dNK(:,j) = W(1,j).^p * reshape(distance(j,:,:),[N,d]) * WBETA ;
end
j_fun = sum(sum(dNK .* transpose(Cluster_elem.^fuzzy_degree)));

end

