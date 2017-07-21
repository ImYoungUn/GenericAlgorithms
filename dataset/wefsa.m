function [opt_fea_vec, g_opt_val, t_opt_val] = wefsa_sp2( IDATA, IANSWER, IPOOL_SIZE, INCALLS, ILCONS, test_data, test_answer)
% Wrapper: Estimation of Distribution Algorithm-based Feature Selection Algorithm
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


t_opt_val = zeros(ncalls, 1);
g_opt_val = zeros(ncalls, 1);

[row,col] = size( data );

% due to multi-label accuracy family, 1+4 cells needs to be saved
% [acalls,primary measures,secondary measures,...,time]

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

[eval,sidx] = sort( eval,1, 'descend');
pool = pool(sidx,:);
g_best_pop = pool(1,:);
g_best_eval = eval(1,:);

%% Start Generation
for m = 1: ncalls % until termination condition is satisfied
    proto = sum( pool(1:round(pool_size/2),:) ) / round(pool_size/2);
    
    % Generate the new population by sampling the estimated distribution
    for k=1:pool_size
        pool(k,:) = 0;
        pool(k,randsample( 1:size(pool,2), 1+round(rand()*49), true, proto ) ) = 1;
        
        eval(k,:) = evaluate( pool(k,:) );
         
    end
    
    [eval,sidx] = sort( eval,1, 'descend');
    pool = pool(sidx,:);

    if eval(1,1) > g_best_eval(1,1)
        g_best_pop = pool(1,:);
        g_best_eval = eval(1,:);
    end
    mdl= ClassificationKNN.fit(data(:, g_best_pop(1,:) == 1,:), answer, 'NumNeighbors', 5);
    pre = mdl.predict(test_data(:, g_best_pop(1,:) == 1));
    g_opt_val(m,1) = sum(pre == test_answer) / size(test_answer, 1);
    t_opt_val(m,1) = g_best_eval(1,1);
end
opt_fea_vec = g_best_pop;
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

