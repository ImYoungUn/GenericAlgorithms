function [opt_fea_vec, g_opt_val, t_opt_val] = wgfsa( IDATA, IANSWER, IPOOL_SIZE, INCALLS, ILCONS, test_data, test_answer)
% Wrapper: Genetic Algorithm-based Feature Selection Algorithm
%
%Input
%   IDATA : Training Data
%   IANSWER : Training Data answer
%   IPOOL_SIZE : Chromosome 개수
%   INCALLS : 최대 Iteration 횟수
%   ILCONS : 선택할 최대 특징 개수
%   test_data : Test data
%   test_answer
%Output
%   opt_fea_ve : Iteration이 끝난 후 최적의 특징 부분집합
%   g_opt_val : Iteration 마다의 테스트 데이터 정확도
%   t_opt_val : Iteration 마다의 트레이닝 데이터 내부의 Croos-validation을 통한 정확도



global data
global answer
global pool_size
global pool
global eval
global ncalls
global col
global row
global lcons

data = IDATA;
answer = IANSWER;
pool_size = IPOOL_SIZE;
ncalls = INCALLS;
lcons = ILCONS;


g_opt_val = zeros(ncalls, 1);
t_opt_val = zeros(ncalls, 1);

[row,col] = size( data );

% due to multi-label accuracy family, 1+4 cells needs to be saved
% [acalls,primary measures,secondary measures,...]
%% Initialize P(t)
% Randomly initialize the pool
% Each chromosome must contain less than LCONS '1' bit
idx = 1:col;
pool = zeros(pool_size,col);
for k=1:pool_size
    tidx = idx;
    rlen = round(rand()*min((lcons-1),(col-1)))+1;
    for m=1:rlen
        ridx = round(rand()*(length(tidx)-1))+1;
        pool(k,tidx(ridx)) = 1;
        tidx(ridx) = [];
    end
end

% for the case of Multi-label accuracy, pre-allocate 4 cells; [mlacc mlprec mlrec mlf1]
% for the other evaluation measures, values are assigned as NaN
eval = zeros(pool_size,4);
eval(:,:) = inf;

%% Evaluate P(t)
for k=1:pool_size
    eval(k,:) = evaluate( pool(k,:) ); % Obtaining each fitness
end

%% LS Improvement P(t) according to the Age
[eval,sidx] = sort( eval,1, 'descend');
pool = pool(sidx,:);

%% Start Generation
for m = 1:ncalls % until termination condition is satisfied
    while size(pool,1) <= pool_size
        % Crossover
        offsprings = crossover();
        c1_eval = evaluate( offsprings(1,:) );
        c2_eval = evaluate( offsprings(2,:) );

        pool = [pool;offsprings];
        eval = [eval;c1_eval;c2_eval];                     
        
        % Mutation
        child = mutation();
        c_eval = evaluate( child );

        pool = [pool;child];
        eval = [eval;c_eval];                  
        
        [pool,sidx] = unique( pool, 'rows', 'first' );        
        eval = eval(sidx,:);
    end
    
    [eval,sidx] = sort( eval,1, 'descend');
    pool = pool(sidx,:);
    eval = eval(1:pool_size,:);
    pool = pool(1:pool_size,:);
   
    mdl= ClassificationKNN.fit(data(:, pool(1,:) == 1,:), answer, 'NumNeighbors', 5);
    pre = mdl.predict(test_data(:, pool(1,:) == 1));
    g_opt_val(m,1) = sum(pre == test_answer) / size(test_answer, 1);
    t_opt_val(m,1) = eval(1,1);
end
opt_fea_vec = pool(1,:);
end


function children = crossover()
% Restrictive Crossover

global pool
global lcons

% Mating the best chromosome p1 and a randomly selected chromosome p2
pidx = round(rand()*(size(pool,1)-1))+1;
midx = round(rand()*(size(pool,1)-1))+1;
while pidx == midx
    midx = round(rand()*(size(pool,1)-1))+1;
end
children = zeros( 2, size(pool,2) );

% Perform the single point crossover
spoint = round(rand()*(size(pool,2)-1))+1;
children(1,1:spoint) = pool(pidx,1:spoint);
children(1,spoint+1:end) = pool(midx,spoint+1:end);
children(2,1:spoint) = pool(midx,1:spoint);
children(2,spoint+1:end) = pool(pidx,spoint+1:end);

% Make sure that the number of selected features after crossover
% obey the number of allowable bit
for k=1:2
    chr_len = sum(children(k,:));
    if chr_len > lcons
        ones_list = find(children(k,:)==1);
        for m=1:chr_len-lcons
            ridx = round(rand()*(length(ones_list)-1))+1;
            children(k,ones_list(ridx)) = 0;
            ones_list(ridx) = [];
        end
    end
end
end

function child = mutation()
% Restrictive Mutation

global pool
global lcons

pidx = round(rand()*(size(pool,1)-1))+1;

child = pool(pidx,:);
for k=1:length(find(pool(pidx,:)==1))
    p_zeroslist = find(child==0);
    
    if ~isempty(p_zeroslist)
        p_oneslist = find(child==1);
        zidx = round(rand()*(length(p_zeroslist)-1))+1;

        child( 1, p_zeroslist(zidx) ) = 1;
        child( 1, p_oneslist(k) ) = 0;
    end
end

for k=1:lcons-length(find(child==1))
    if rand() < 0.1 % Mutation Rate
        p_zeroslist = find(child==0);
        
        if ~isempty(p_zeroslist)
            zidx = round(rand()*(length(p_zeroslist)-1))+1;
            child( 1, p_zeroslist(zidx) ) = 1;
        end
    end
end
end



function val = evaluate( chr )


% Increase the number of actual fitness function calls


if all(chr==0)
    val = [inf NaN NaN NaN];
    return;
end
val = zeros(1,4);
global data
global row
global answer

[train,test] = crossvalind( 'holdout', ones(row,1), 0.2 );
mdl = ClassificationKNN.fit(data(train, chr == 1), answer(train, :), 'NumNeighbors', 5);
pre = mdl.predict(data(test, chr==1));
val(1,1) = sum(pre == answer(test, :)) / size(answer(test, :), 1);
end



