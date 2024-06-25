
function D = spectralReconstruction(Dtot, Dtot_masked, mask)
%%
% D = spectralReconstruction(Dtot, Dmasked, mask)
%
% Completes a partially observed D using Soft-impute,
% an iterative soft-thresholded svds to impute the missing values.
%
% INPUT:  Dtot ... original, complete matrix, needed to compute best parameter lambda
%         Dtot_masked...incomplete observed matrix
%         mask ... observation matrix (0 non observed value, 1 observed)
%
% OUTPUT: D   ... completed EDM
%
% From pseudocode from
% Mazumder R, Hastie T, Tibshirani R. 
% Spectral Regularization Algorithms for Learning Large Incomplete Matrices. 
% J Mach Learn Res. 2010 Mar 1;11:2287-2322. PMID: 21552465; PMCID: PMC3087301.

%%%% SEARCHING BEST REGULARISATION PARAMETER %%%%

lambdav = 10.^(linspace(-4,2,20));

%%%% PARAMETERS %%%%

NitMax = 100000;
toll = 1e-5;


Zold = zeros(size(Dtot));


I = mask;
X = Dtot;
X_mask = Dtot_masked;


PX = X_mask;

ev = [];
e2 = norm(Zold-X)/norm(X)*100;


for j=1:length(lambdav)
    
    lambda = lambdav(j);

    for i=1:NitMax
        
        PZold = Zold-Zold.*I;
        A = PX+PZold;
        [U,S,V] = svd(A);
        SS = diag(max(0,diag(S)-lambda));
        Znew = U*SS*V';       
        e = norm(Znew-Zold, "fro")/(norm(Zold, "fro")+1e-20);
        if (e<toll)
            break
        end
        D =Zold;
        Zold = Znew;
    
    end
    e2new = norm(Znew-X)/norm(X)*100;

    if e2new < e2
        e2 = e2new;
    elseif e2new >= e2
        break
    end

    %ev = [ev e2new];

end
%isequal(Zoldold, Zold)

%plot(ev)