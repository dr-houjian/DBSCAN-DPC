%This is the implementation of the DBSCAN-DPC algorithm proposed in
%Jian Hou, Houshen Lin, Huaqiang Yuan, Marcello Pelillo. Flexible Density
%Peak Clustering for Real-World Data. Pattern Recognition, vol. 156, 110772, December2024

function demo_dbscan_dpc()

    nk=11;
    nsample0=0.028;
    minpts=4;

    fname='dataset\usps-2d-tsne.txt';
    
    descr=dlmread(fname);
    dimen=size(descr,2);
    label_t=descr(:,dimen);
    descr=descr(:,1:dimen-1);
    ndata=size(descr,1);
    nsample=max(round(ndata*nsample0),5);

    %DP related
    rho=zeros(1,ndata);
    dima0=pdist(descr,'euclidean');
    dima=squareform(dima0);
    dmax=max(dima(:));
    for ii=1:ndata
        vec=dima(ii,:);
        vec1=sort(vec,'ascend');
        rho(ii)=dmax/mean(vec1(2:nsample+1)); 
    end
    [~,ordrho,nneigh]=find_delta(rho,dima);
    
    %start clustering
    label_c=zeros(1,ndata);

    no=1;
    while 1>0
        if sum(label_c==0)<5
            break;
        end
        
        %initial cluster
        label_c=iniclu_dbscan(dima,label_c,ordrho,no,minpts,nk);

        %expansion by dp
        label_c=cluster_extend_dp(label_c,no,dima,nneigh,rho);

        no=no+1;
    end
    
    cq=label2cq(label_c,label_t);
    rate_nmi=cq.nmi
    rate_acc=cq.acc

end

function label=iniclu_dbscan(dima,label,ordgamma,no_cluster,minpts,nk)

    ndata=length(label);
    
    idx_seed=0;
    for i=1:ndata
        if label(ordgamma(i))==0
            idx_seed=ordgamma(i);
            break;
        end
    end

    vec_d=dima(idx_seed,:);
    vec_d1=sort(vec_d,'ascend');
    eps=vec_d1(nk+1);
    
    ndata=size(dima,1);
    touched=zeros(1,ndata);
    
    %clustering starting from the seed
    touched(idx_seed)=1;
    label(idx_seed)=no_cluster;
    
    vec_d=dima(idx_seed,:);
    ind=find(vec_d<=eps);
    
    while ~isempty(ind)
        idx=ind(1);
        ind(1)=[];
            
        if touched(idx)==0
            touched(idx)=1;
                
            dist=dima(idx,:);
            i1=find(dist<=eps);
                
            if length(i1)>=minpts
                ind=[ind i1];
                ind=unique(ind);
            end
        end
        
        if label(idx)==0
            label(idx)=no_cluster;
        end
    end

end

function label=cluster_extend_dp(label,num_dsets,dima,nneigh,rho)

    while 1>0
        idx_ndset=find(label==0);
        idx_dset= label==num_dsets;
        sub_dima=dima(idx_ndset,idx_dset);
        [~,idx_min]=min(sub_dima,[],1);
        idx_out=idx_ndset(idx_min);
        idx_out=unique(idx_out);
                
        rho0=rho(idx_out);
        [~,idx_out1]=sort(rho0,'descend');

        flag=0;
        for i=idx_out(idx_out1)   %sort based on rho
            idx_neigh=nneigh(i);
                        
            if idx_neigh==0
                continue;
            end
            
            if label(idx_neigh)==num_dsets
                label(i)=num_dsets;
                flag=1;
            end
        end
        
        if flag==0
            break;
        end
    end
end

function [delta,ordrho,nneigh]=find_delta(rho,dist)

    ND=length(rho);
    delta=zeros(1,ND);
    nneigh=zeros(1,ND);
    
    maxd=max(max(dist));
    
    [~,ordrho]=sort(rho,'descend');
    delta(ordrho(1))=-1;
    nneigh(ordrho(1))=0;
    
    for i=2:ND
        delta(ordrho(i))=maxd+1;
        for j=1:i-1
            if(dist(ordrho(i),ordrho(j))<delta(ordrho(i)))
                delta(ordrho(i))=dist(ordrho(i),ordrho(j));
                nneigh(ordrho(i))=ordrho(j);
            end
        end
    end
    delta(ordrho(1))=max(delta(:));
    
end


function cq=label2cq(label_c,label_t)

    min_labelc=min(label_c);
    if min_labelc<1
        label_c=label_c+1-min_labelc;
    end
    min_labelt=min(label_t);
    if min_labelt<1
        label_t=label_t+1-min_labelt;
    end

    label_c=reshape(label_c,size(label_t));
    
    cq.acc=ClusteringAccuracy(label_c,label_t);
    cq.nmi=nmi_adopt(label_c,label_t);

end

function Acc = ClusteringAccuracy(label,rlabel)
    % %  Clustering Accuracy
    % % Deng Cai, Xiaofei He, and Jiawei Han,
    %"Document Clustering Using Locality Preserving Indexing", in IEEE TKDE, 2005. 

    % two subcode
    % bestMap, hungarian

    % Input
    % label    real label
    % rlabel   cluster label

    % Output
    % Acc      Accuracy

    res = bestMap(label,rlabel);
    Acc = length(find(label == res))/length(label);

    clear res;
end

function [newL2] = bestMap(L1,L2)
    %bestmap: permute labels of L2 to match L1 as good as possible
    %   [newL2] = bestMap(L1,L2);
    %
    %   version 2.0 --May/2007
    %   version 1.0 --November/2003
    %
    %   Written by Deng Cai (dengcai AT gmail.com)


    %===========    

    L1 = L1(:);
    L2 = L2(:);
    if size(L1) ~= size(L2)
        error('size(L1) must == size(L2)');
    end

    Label1 = unique(L1);
    nClass1 = length(Label1);
    Label2 = unique(L2);
    nClass2 = length(Label2);

    nClass = max(nClass1,nClass2);
    G = zeros(nClass);
    for i=1:nClass1
        for j=1:nClass2
            G(i,j) = length(find(L1 == Label1(i) & L2 == Label2(j)));
        end
    end

    [c,t] = hungarian(-G);
    newL2 = zeros(size(L2));
    for i=1:nClass2
        if c(i) > nClass1
            continue;
        end
        newL2(L2 == Label2(i)) = Label1(c(i));
    end

    clear G;
end

function [C,T]=hungarian(A)
    %HUNGARIAN Solve the Assignment problem using the Hungarian method.
    %
    %[C,T]=hungarian(A)
    %A - a square cost matrix.
    %C - the optimal assignment.
    %T - the cost of the optimal assignment.
    %s.t. T = trace(A(C,:)) is minimized over all possible assignments.

    % Adapted from the FORTRAN IV code in Carpaneto and Toth, "Algorithm 548:
    % Solution of the assignment problem [H]", ACM Transactions on
    % Mathematical Software, 6(1):104-111, 1980.

    % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.
    %                 Department of Computing Science, Ume?University,
    %                 Sweden. 
    %                 All standard disclaimers apply.

    % A substantial effort was put into this code. If you use it for a
    % publication or otherwise, please include an acknowledgement or at least
    % notify me by email. /Niclas

    [m,n]=size(A);

    if (m~=n)
        error('HUNGARIAN: Cost matrix must be square!');
    end

    % Save original cost matrix.
    orig=A;

    % Reduce matrix.
    A=hminired(A);

    % Do an initial assignment.
    [A,C,U]=hminiass(A);

    % Repeat while we have unassigned rows.
    while (U(n+1))
        % Start with no path, no unchecked zeros, and no unexplored rows.
        LR=zeros(1,n);
        LC=zeros(1,n);
        CH=zeros(1,n);
        RH=[zeros(1,n) -1];

        % No labelled columns.
        SLC=[];

        % Start path in first unassigned row.
        r=U(n+1);
        % Mark row with end-of-path label.
        LR(r)=-1;
        % Insert row first in labelled row set.
        SLR=r;

        % Repeat until we manage to find an assignable zero.
        while (1)
            % If there are free zeros in row r
            if (A(r,n+1)~=0)
                % ...get column of first free zero.
                l=-A(r,n+1);

                % If there are more free zeros in row r and row r in not
                % yet marked as unexplored..
                if (A(r,l)~=0 & RH(r)==0)
                    % Insert row r first in unexplored list.
                    RH(r)=RH(n+1);
                    RH(n+1)=r;

                    % Mark in which column the next unexplored zero in this row
                    % is.
                    CH(r)=-A(r,l);
                end
            else
                % If all rows are explored..
                if (RH(n+1)<=0)
                    % Reduce matrix.
                    [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
                end

                % Re-start with first unexplored row.
                r=RH(n+1);
                % Get column of next free zero in row r.
                l=CH(r);
                % Advance "column of next free zero".
                CH(r)=-A(r,l);
                % If this zero is last in the list..
                if (A(r,l)==0)
                    % ...remove row r from unexplored list.
                    RH(n+1)=RH(r);
                    RH(r)=0;
                end
            end

            % While the column l is labelled, i.e. in path.
            while (LC(l)~=0)
                % If row r is explored..
                if (RH(r)==0)
                    % If all rows are explored..
                    if (RH(n+1)<=0)
                        % Reduce cost matrix.
                        [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
                    end

                    % Re-start with first unexplored row.
                    r=RH(n+1);
                end

                % Get column of next free zero in row r.
                l=CH(r);

                % Advance "column of next free zero".
                CH(r)=-A(r,l);

                % If this zero is last in list..
                if(A(r,l)==0)
                    % ...remove row r from unexplored list.
                    RH(n+1)=RH(r);
                    RH(r)=0;
                end
            end

            % If the column found is unassigned..
            if (C(l)==0)
                % Flip all zeros along the path in LR,LC.
                [A,C,U]=hmflip(A,C,LC,LR,U,l,r);
                % ...and exit to continue with next unassigned row.
                break;
            else
                % ...else add zero to path.

                % Label column l with row r.
                LC(l)=r;

                % Add l to the set of labelled columns.
                SLC=[SLC l];

                % Continue with the row assigned to column l.
                r=C(l);

                % Label row r with column l.
                LR(r)=l;

                % Add r to the set of labelled rows.
                SLR=[SLR r];
            end
        end
    end

    % Calculate the total cost.
    T=sum(orig(logical(sparse(C,1:size(orig,2),1))));

end

function A=hminired(A)
    %HMINIRED Initial reduction of cost matrix for the Hungarian method.
    %
    %B=assredin(A)
    %A - the unreduced cost matris.
    %B - the reduced cost matrix with linked zeros in each row.

    % v1.0  96-06-13. Niclas Borlin, niclas@cs.umu.se.

    [m,n]=size(A);

    % Subtract column-minimum values from each column.
    colMin=min(A);
    A=A-colMin(ones(n,1),:);

    % Subtract row-minimum values from each row.
    rowMin=min(A')';
    A=A-rowMin(:,ones(1,n));

    % Get positions of all zeros.
    [i,j]=find(A==0);

    % Extend A to give room for row zero list header column.
    A(1,n+1)=0;
    for k=1:n
        % Get all column in this row. 
        cols=j(k==i)';
        % Insert pointers in matrix.
        A(k,[n+1 cols])=[-cols 0];
    end
end

function [A,C,U]=hminiass(A)
    %HMINIASS Initial assignment of the Hungarian method.
    %
    %[B,C,U]=hminiass(A)
    %A - the reduced cost matrix.
    %B - the reduced cost matrix, with assigned zeros removed from lists.
    %C - a vector. C(J)=I means row I is assigned to column J,
    %              i.e. there is an assigned zero in position I,J.
    %U - a vector with a linked list of unassigned rows.

    % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    [n,np1]=size(A);

    % Initalize return vectors.
    C=zeros(1,n);
    U=zeros(1,n+1);

    % Initialize last/next zero "pointers".
    LZ=zeros(1,n);
    NZ=zeros(1,n);

    for i=1:n
        % Set j to first unassigned zero in row i.
        lj=n+1;
        j=-A(i,lj);

        % Repeat until we have no more zeros (j==0) or we find a zero
        % in an unassigned column (c(j)==0).

        while (C(j)~=0)
            % Advance lj and j in zero list.
            lj=j;
            j=-A(i,lj);

            % Stop if we hit end of list.
            if (j==0)
                break;
            end
        end

        if (j~=0)
            % We found a zero in an unassigned column.

            % Assign row i to column j.
            C(j)=i;

            % Remove A(i,j) from unassigned zero list.
            A(i,lj)=A(i,j);

            % Update next/last unassigned zero pointers.
            NZ(i)=-A(i,j);
            LZ(i)=lj;

            % Indicate A(i,j) is an assigned zero.
            A(i,j)=0;
        else
            % We found no zero in an unassigned column.

            % Check all zeros in this row.

            lj=n+1;
            j=-A(i,lj);

            % Check all zeros in this row for a suitable zero in another row.
            while (j~=0)
                % Check the in the row assigned to this column.
                r=C(j);

                % Pick up last/next pointers.
                lm=LZ(r);
                m=NZ(r);

                % Check all unchecked zeros in free list of this row.
                while (m~=0)
                    % Stop if we find an unassigned column.
                    if (C(m)==0)
                        break;
                    end

                    % Advance one step in list.
                    lm=m;
                    m=-A(r,lm);
                end

                if (m==0)
                    % We failed on row r. Continue with next zero on row i.
                    lj=j;
                    j=-A(i,lj);
                else
                    % We found a zero in an unassigned column.

                    % Replace zero at (r,m) in unassigned list with zero at (r,j)
                    A(r,lm)=-j;
                    A(r,j)=A(r,m);

                    % Update last/next pointers in row r.
                    NZ(r)=-A(r,m);
                    LZ(r)=j;

                    % Mark A(r,m) as an assigned zero in the matrix . . .
                    A(r,m)=0;

                    % ...and in the assignment vector.
                    C(m)=r;

                    % Remove A(i,j) from unassigned list.
                    A(i,lj)=A(i,j);

                    % Update last/next pointers in row r.
                    NZ(i)=-A(i,j);
                    LZ(i)=lj;

                    % Mark A(r,m) as an assigned zero in the matrix . . .
                    A(i,j)=0;

                    % ...and in the assignment vector.
                    C(j)=i;

                    % Stop search.
                    break;
                end
            end
        end
    end

    % Create vector with list of unassigned rows.

    % Mark all rows have assignment.
    r=zeros(1,n);
    rows=C(C~=0);
    r(rows)=rows;
    empty=find(r==0);

    % Create vector with linked list of unassigned rows.
    U=zeros(1,n+1);
    U([n+1 empty])=[empty 0];
end

function [A,C,U]=hmflip(A,C,LC,LR,U,l,r)
    %HMFLIP Flip assignment state of all zeros along a path.
    %
    %[A,C,U]=hmflip(A,C,LC,LR,U,l,r)
    %Input:
    %A   - the cost matrix.
    %C   - the assignment vector.
    %LC  - the column label vector.
    %LR  - the row label vector.
    %U   - the 
    %r,l - position of last zero in path.
    %Output:
    %A   - updated cost matrix.
    %C   - updated assignment vector.
    %U   - updated unassigned row list vector.

    % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    n=size(A,1);

    while (1)
        % Move assignment in column l to row r.
        C(l)=r;

        % Find zero to be removed from zero list..

        % Find zero before this.
        m=find(A(r,:)==-l);

        % Link past this zero.
        A(r,m)=A(r,l);

        A(r,l)=0;

        % If this was the first zero of the path..
        if (LR(r)<0)
            ...remove row from unassigned row list and return.
            U(n+1)=U(r);
            U(r)=0;
            return;
        else

            % Move back in this row along the path and get column of next zero.
            l=LR(r);

            % Insert zero at (r,l) first in zero list.
            A(r,l)=A(r,n+1);
            A(r,n+1)=-l;

            % Continue back along the column to get row of next zero in path.
            r=LC(l);
        end
    end

end

function [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
    %HMREDUCE Reduce parts of cost matrix in the Hungerian method.
    %
    %[A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
    %Input:
    %A   - Cost matrix.
    %CH  - vector of column of 'next zeros' in each row.
    %RH  - vector with list of unexplored rows.
    %LC  - column labels.
    %RC  - row labels.
    %SLC - set of column labels.
    %SLR - set of row labels.
    %
    %Output:
    %A   - Reduced cost matrix.
    %CH  - Updated vector of 'next zeros' in each row.
    %RH  - Updated vector of unexplored rows.

    % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    n=size(A,1);

    % Find which rows are covered, i.e. unlabelled.
    coveredRows=LR==0;

    % Find which columns are covered, i.e. labelled.
    coveredCols=LC~=0;

    r=find(~coveredRows);
    c=find(~coveredCols);

    % Get minimum of uncovered elements.
    m=min(min(A(r,c)));

    % Subtract minimum from all uncovered elements.
    A(r,c)=A(r,c)-m;

    % Check all uncovered columns..
    for j=c
        % ...and uncovered rows in path order..
        for i=SLR
            % If this is a (new) zero..
            if (A(i,j)==0)
                % If the row is not in unexplored list..
                if (RH(i)==0)
                    % ...insert it first in unexplored list.
                    RH(i)=RH(n+1);
                    RH(n+1)=i;
                    % Mark this zero as "next free" in this row.
                    CH(i)=j;
                end
                % Find last unassigned zero on row I.
                row=A(i,:);
                colsInList=-row(row<0);
                if (length(colsInList)==0)
                    % No zeros in the list.
                    l=n+1;
                else
                    l=colsInList(row(colsInList)==0);
                end
                % Append this zero to end of list.
                A(i,l)=-j;
            end
        end
    end

    % Add minimum to all doubly covered elements.
    r=find(coveredRows);
    c=find(coveredCols);

    % Take care of the zeros we will remove.
    [i,j]=find(A(r,c)<=0);

    i=r(i);
    j=c(j);

    for k=1:length(i)
        % Find zero before this in this row.
        lj=find(A(i(k),:)==-j(k));
        % Link past it.
        A(i(k),lj)=A(i(k),j(k));
        % Mark it as assigned.
        A(i(k),j(k))=0;
    end

    A(r,c)=A(r,c)+m;

end


function z = nmi_adopt(x, y)
    % Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
    % Input:
    %   x, y: two integer vector of the same length 
    % Ouput:
    %   z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
    % Written by Mo Chen (sth4nth@gmail.com).
    assert(numel(x) == numel(y));
    n = numel(x);
    x = reshape(x,1,n);
    y = reshape(y,1,n);

    l = min(min(x),min(y));
    x = x-l+1;
    y = y-l+1;
    k = max(max(x),max(y));

    idx = 1:n;
    Mx = sparse(idx,x,1,n,k,n);
    My = sparse(idx,y,1,n,k,n);
    Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
    Hxy = -dot(Pxy,log2(Pxy));


    % hacking, to elimative the 0log0 issue
    Px = nonzeros(mean(Mx,1));
    Py = nonzeros(mean(My,1));

    % entropy of Py and Px
    Hx = -dot(Px,log2(Px));
    Hy = -dot(Py,log2(Py));

    % mutual information
    MI = Hx + Hy - Hxy;

    % normalized mutual information
    z = sqrt((MI/Hx)*(MI/Hy));
    z = max(0,z);

    clear Hx Hy MI;
end
