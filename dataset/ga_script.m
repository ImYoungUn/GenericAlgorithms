%���� �ڵ� Genetic Algorithm

clear;
load('Isolet.mat');
sim_seq = SeqGen(size(X,2),size(X,1),0.2);
rep_size = 20; %���� �ݺ�Ƚ��
iteration_size = 50; %wrapper ���� iteration�� ��� ���� ������
population_size = 50; %chromosome�� ��� �Ұ�����
max_feature_size = 50; %�ִ� feature ���� ������ ��� �� ������

g_opt_table = zeros(iteration_size, rep_size);


for k = 1:rep_size
    train_data = X(sim_seq(:,k), :);
    train_answer = Y(sim_seq(:,k), :);
    test_data = X(~sim_seq(:,k), :);
    test_answer = Y(~sim_seq(:,k), :);
    
    [~, g_opt_table(:, k), ~] = wpfsa( train_data, train_answer, population_size, iteration_size, max_feature_size, test_data, test_answer);  
    save('Isolet_result2.mat', 'g_opt_table');
end

mean_acc = mean(g_opt_table(iteration_size, :)); % ������ iteration������ test_data ��Ȯ���� �� ���� ���� ���ؼ� ����� ��
std_acc = std(g_opt_table(iteration_size, :));
save('Isolet_result2', 'g_opt_table', 'mean_acc', 'std_acc'); %������ ����

%mean_acc�� �ᱹ �� Ư¡ ���� �˰������� ������ �����ִ� ô���� ��

