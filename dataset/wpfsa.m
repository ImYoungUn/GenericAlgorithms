function [opt_fea_vec, g_opt_val, t_opt_val] = wpfsa_sp2( IDATA, IANSWER, IPOOL_SIZE, INCALLS, ILCONS, test_data, test_answer)
% Wrapper: Particle swarm optimization-based Feature Selection Algorithm
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
global acalls

g_opt_val = zeros(ncalls, 1);
t_opt_val = zeros(ncalls, 1);

data = IDATA;
answer = IANSWER;
pool_size = IPOOL_SIZE;
ncalls = INCALLS;
acalls = 0;
lcons = ILCONS;


[row,col] = size( data );

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


%% Create individual, global best memory, and velocity
p_best_pop = pool;
p_best_eval = eval;
[~,tidx] = max(eval(:,1));
g_best_pop = pool(tidx,:);
g_best_eval = eval(tidx,:);
velocity = zeros( size( pool ) ); % Initial velocity is set to zero

%% Start Generation
c_1 = 2; c_2 = 2;
v_max = 6; v_min = -6;
w_max = 0.9; w_min = 0.4;
for iter = 1:ncalls % until termination condition is satisfied
    % Obtain the value of w
    w = w_max - (w_max-w_min)*acalls / ncalls;
    for k=1:pool_size
        % Obtain the value of v
        velocity(k,:) = w * velocity(k,:) ...
            + c_1 * rand() * (p_best_pop(k,:) - pool(k,:)) ...
            + c_2 * rand() * (g_best_pop - pool(k,:));
        velocity(k, velocity(k,:) > v_max) = v_max;
        velocity(k, velocity(k,:) < v_min) = v_min;
        
        for m=1:size(pool,2)
            % Obtain threshold: s_{ij}
            if rand() < 1 / (1 + exp( velocity(k,m) ) )
                pool(k,m) = 1;
            else
                pool(k,m) = 0;
            end
        end
        
        % To constrain the bit of '1' into 'lcons'
        one_bit = sum(pool(k,:));
        if one_bit > lcons
            tidx = randsample( find( pool(k,:) == 1 ), one_bit - lcons );
            pool(k,tidx) = 0;
        end
        
        eval(k,:) = evaluate( pool(k,:) );
        if eval(k,1) > p_best_eval(k,1)
            p_best_pop(k,:) = pool(k,:);
            p_best_eval(k,:) = eval(k,:);
        end
    end
    [~,tidx] = max(eval(:,1));
    if eval(tidx,1) > g_best_eval
        g_best_pop = pool(tidx,:);
        g_best_eval = eval(tidx,:);
    end
    mdl= ClassificationKNN.fit(data(:, g_best_pop(1,:) == 1,:), answer, 'NumNeighbors', 5);
    pre = mdl.predict(test_data(:, g_best_pop(1,:) == 1));
    g_opt_val(iter,1) = sum(pre == test_answer) / size(test_answer, 1);
    t_opt_val(iter,1) = g_best_eval(1,1);
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



