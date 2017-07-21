%예시 코드 Genetic Algorithm

clear;
load('Isolet.mat');
sim_seq = SeqGen(size(X,2),size(X,1),0.2);
rep_size = 20; %실험 반복횟수
iteration_size = 50; %wrapper 내에 iteration을 몇번 돌릴 것인지
population_size = 50; %chromosome을 몇개로 할것인지
max_feature_size = 50; %최대 feature 선택 개수를 몇개로 할 것인지

g_opt_table = zeros(iteration_size, rep_size);


for k = 1:rep_size
    train_data = X(sim_seq(:,k), :);
    train_answer = Y(sim_seq(:,k), :);
    test_data = X(~sim_seq(:,k), :);
    test_answer = Y(~sim_seq(:,k), :);
    
    [~, g_opt_table(:, k), ~] = wpfsa( train_data, train_answer, population_size, iteration_size, max_feature_size, test_data, test_answer);  
    save('Isolet_result2.mat', 'g_opt_table');
end

mean_acc = mean(g_opt_table(iteration_size, :)); % 마지막 iteration에서의 test_data 정확도를 각 실험 마다 구해서 평균을 냄
std_acc = std(g_opt_table(iteration_size, :));
save('Isolet_result2', 'g_opt_table', 'mean_acc', 'std_acc'); %데이터 저장

%mean_acc가 결국 그 특징 선택 알고리즘의 성능을 구해주는 척도가 됨


