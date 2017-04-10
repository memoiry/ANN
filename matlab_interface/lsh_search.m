function [result, dists] = lsh_search(dataset, query, k_num, table_num, key_size)
    %pass the parameter and query to python
    fvecs_write('dataset.fvecs',dataset)
    fvecs_write('query.fvecs',query);
    dos(sprintf('python python_nn_search.py %d %d %d', k_num, table_num, key_size));
    %waiting untile python finished computing.
    delete('result.csv');
    while exist('result.csv','file') ~= 2
    end
    %read the result
    result = csvread('result.csv');
    dists = csvread('dists.csv');
    %delete in case reloading.
    delete('result.csv');
    delete('dists.csv');
end
