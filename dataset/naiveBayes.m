function res = naiveBayes(X, Y, sim_seq)

sim_num = size(sim_seq, 2);
res = zeros(sim_num, 1);


for i = 1:sim_num
    idx = sim_seq(:,i);
    
    tr_data = X(idx,:);
    tr_ans = Y(idx,:);
    ts_data = X(~idx,:);
    ts_ans = Y(~idx,:);
    
    mdl = NaiveBayes.fit(tr_data ,tr_ans, 'Distribution', 'normal');
    pre = mdl.predict(ts_data);
    
    res(i,1) = sum(pre == ts_ans) / size(ts_ans, 1);
    
end

end