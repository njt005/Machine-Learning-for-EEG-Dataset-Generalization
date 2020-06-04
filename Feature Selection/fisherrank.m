%****************************************************************************************************
%
%   Ranking features by using fisher criterion
%
%   Author: Stefan Ehrlich
%   Last revised: 15.12.2014
%
%   Input:
%   - featv: feature vector [samples, features]
%   - labels
%   Output:
%   - d: fisher criterion
%   - rank: descending rank of features per discriminative quality
%
%****************************************************************************************************

function [d, rank, score] = fisherrank(featv,labels)

    nfeat = size(featv,2);
    c1 = featv(labels==-1,:);
    c2 = featv(labels==1,:);
    
    for f=1:nfeat
        d(f) = (mean(c1(:,f))-mean(c2(:,f)))^2/(var(c1(:,f))+var(c2(:,f)));
    end
    
    score = d;
    [d, rank] = sort(d,'descend');

end