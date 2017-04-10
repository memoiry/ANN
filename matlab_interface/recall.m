function acc = recall(result, ground_truth, k)
    num = size(result,1);
    dim = size(result,2);
    size_ = num*dim;
    count = 0;
    for i = 1:num
        temp = result(i,1:k);
        temp_truth = ground_truth(i,1:k);
        for j = 1:dim
            count = count + any(temp(j) == temp_truth);
        end
    end
    acc = count / size_;
end