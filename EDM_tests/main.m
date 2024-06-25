function [err_mc, err_std_mc, rel_err_mc, rel_err_std_mc, err_DX2_mc, err_DX2_sd_mc, rel_err_DX2_mc, rel_err_DX2_sd_mc]  = main()

% %%%%%%%% DATASET %%%%%%%%
% 
% Load the Fisher's Iris dataset:
% 150 specimens, 4 features
%     
% dataset = 'Iris';
% load fisheriris meas;
% X = meas;
% 
% % Perform Min-Max scaling on X
% min_X = min(X);
% max_X = max(X);
% X = (X - min_X) ./ (max_X - min_X);
%     
%--------------------------------------------------------------

% dataset = 'Ionosphere';
% load ionosphere.mat X;
% 
% X = normalize(X, 'range', [0,1]);

%--------------------------------------------------------------

dataset = 'Sonar';

file_name = 'sonar.all-data.csv'; 
file_path = fullfile(pwd, file_name); % Full path to the dataset file
full_data = readmatrix(file_path); % Read the entire CSV file
data = full_data(:, 1:end-1); % Exclude the last column (label column)
disp(size(data));
X = normalize(data, 'range', [0,1]);


%--------------------------------------------------------------

    % Matrix completion methods
    methods = {'Alternating_Descent', ...
           'Rank_Alternation', ...
           'Soft-Impute', ...
           'OptSpace',  ...
           'SVT', 'sdr'};

    %%%%%%%% SETTING THE EXPERIMENT %%%%%%%%
    
    [err_mc, err_std_mc, rel_err_mc, rel_err_std_mc,  err_DX2_mc, err_DX2_sd_mc, rel_err_DX2_mc, rel_err_DX2_sd_mc] =...
        Comparing_methods_fc(X, 'subsampling', 'Soft-Impute', dataset);
        % Nystrom method: 'subsampling', 'normal', 'uniform'

end

%%%%%%%% FUNCTIONS %%%%%%%%

function [err_mc, err_std_mc, rel_err_mc, rel_err_std_mc,  err_DX2_mc, err_DX2_sd_mc, rel_err_DX2_mc, rel_err_DX2_sd_mc] = Comparing_methods_fc(X, nyst_method, completion_method, dataset_name)

% INPUT:  X ... samples of the dataset
%         nyst_method ...can be 'uniform', 'normal' or 'subsampling'. Decides how the matrix of Nystrom poiints W is generated. 
%         completion_method ...'Alternating_Descent', 'Rank_Alternation', 'Soft-Impute', 'OptSpace', 'SVT'
%         dataset name ... will be used to create a subfolder with results
% OUTPUT: errors and standard deviations lists


    % Ensure the dataset/method subfolder exists or create it
    folder = fullfile(dataset_name, completion_method, nyst_method);  % Create subfolder for the dataset, completion method and nystrom method
    if ~isfolder(folder)
        mkdir(folder);
    end
    
    fprintf('\n Dataset:  %s \n', dataset_name);
    fprintf('\n Nystrom method:  %s \n', nyst_method);
    fprintf('\n Completion method:  %s \n', completion_method); 

    n = size(X, 1); % Number of points in the set
    d = size(X, 2); % Embedding dimension

    % Minimal and maximal number of Nystrom points 
    nys_min  = 10;
    nys_step = 10;
    nys_max  = min(n, n - mod(n, nys_step));
    %nys_max = 10; % for testing
    nys_array = nys_min:nys_step:nys_max;

    % Number of Monte Carlo trials
    %num_trials = 1; % for testing
    num_trials = 10;

    % Run the Monte Carlo simulation
    [err_mc, err_std_mc, rel_err_mc, rel_err_std_mc, err_DX2_mc, err_DX2_sd_mc, rel_err_DX2_mc, rel_err_DX2_sd_mc] = run_monte_carlo_simulation(X, nys_array, num_trials, nys_min, nys_max, n, d, nyst_method, completion_method, folder, dataset_name);

end




function [err, err_sd, rel_err, rel_err_sd, err_DX2,err_DX2_sd, rel_err_DX2,rel_err_DX2_sd] = run_single_nystrom_trial(X, Mnys, num_trials, n, d, nyst_method, completion_method)
    % Initialize arrays to store results for each method
    err_in = zeros(1, num_trials);
    rel_err_in = zeros(1, num_trials);
    err_in_D_X = zeros(1, num_trials); 
    rel_err_in_D_X = zeros(1, num_trials);

    for trial = 1:num_trials
        rng(trial); % Using the trial number as the seed

        % Generate a new random matrix W for each trial
        if strcmp(nyst_method, 'uniform')
            W = rand(Mnys, size(X, 2));
        elseif strcmp(nyst_method, 'normal')
            W = 1/2 + randn(Mnys, size(X, 2));
        elseif strcmp(nyst_method, 'subsampling') 
            idx = randperm(n, Mnys);
            W = X(idx, :);
        end

        % Nystrom matrix of random points
        D = pdist2(X, W);
        D_X = pdist2(X, X);
        D_X2 = D_X.^2;
        D_W = pdist2(W, W);

        % stack matrices to get 
        % D_X | D
        % -----------
        % D^T | D_W

        D_XD = [D_X, D];
        DTD_W = [D', D_W];

        % composite matrix
        Dtot = [D_XD; DTD_W];
        Dtot = Dtot.^2;

        r = rank(Dtot);


        % Apply the methods to 
        % 0   | D
        % -----------
        % D^T | D_W

        maskDX = zeros(n);
        maskD = ones(n, Mnys);
        maskDT = ones(Mnys, n);
        maskW = ones(Mnys);
        
        % The mask
        mask = [maskDX, maskD; maskDT, maskW]; 

        % The masked total matrix
        Dtot_masked = Dtot .* mask;

        if strcmp(completion_method, 'Alternating_Descent')
            % Applying Alternating Descent
            [~, edm1] = alternating_descent(Dtot_masked, d);
            edm1_D_X2 = edm1(1:n, 1:n);

    
            % Calculate error for Alternating Descent
            err_in(1, trial) = norm(edm1 - Dtot, 'fro');
            rel_err_in(1, trial) = norm(edm1 - Dtot, 'fro') / norm(Dtot, 'fro');

            % Compute the reconstruction error between the submatrices
            err_in_D_X(1, trial) = norm(edm1_D_X2 - D_X2, 'fro');
            rel_err_in_D_X(1,trial) = norm(edm1_D_X2 - D_X2, 'fro')/ norm(D_X2, 'fro');




        elseif strcmp(completion_method, 'Rank_Alternation')
            % Applying Rank Alternation
            edm2 = rank_complete_edm(Dtot_masked, mask, d, 0);
            edm2_D_X2 = edm2(1:n, 1:n);

            % Calculate error for Rank Alternation
            err_in(1, trial) = norm(edm2 - Dtot, 'fro');
            rel_err_in(1, trial) = norm(edm2 - Dtot, 'fro') / norm(Dtot, 'fro');
                        
            % Compute the reconstruction error between the submatrices
            err_in_D_X(1, trial) = norm(edm2_D_X2 - D_X2, 'fro');
            rel_err_in_D_X(1,trial) = norm(edm2_D_X2 - D_X2, 'fro')/ norm(D_X2, 'fro');

        elseif strcmp(completion_method, 'Soft-Impute')
            % Applying Soft-Impute soft-thresholded svds
            edm3 = spectralReconstruction(Dtot, Dtot_masked, mask);
            edm3_D_X2 = edm3(1:n, 1:n);


            % Calculate error for Soft-Impute
            err_in(1, trial) = norm(edm3 - Dtot, 'fro');
            rel_err_in(1, trial) = norm(edm3 - Dtot, 'fro') / norm(Dtot, 'fro');
                        
            % Compute the reconstruction error between the submatrices
            err_in_D_X(1, trial) = norm(edm3_D_X2 - D_X2, 'fro');
            rel_err_in_D_X(1,trial) = norm(edm3_D_X2 - D_X2, 'fro')/ norm(D_X2, 'fro');

        elseif strcmp(completion_method, 'OptSpace')
            % Applying OptSpace
            [edm4, ~] = OptSpace(Dtot_masked, r, [], []);
            edm4_D_X2 = edm4(1:n, 1:n);

    
            % Calculate error for OptSpace
            err_in(1, trial) = norm(edm4 - Dtot, 'fro');
            rel_err_in(1, trial) = norm(edm4 - Dtot, 'fro') / norm(Dtot, 'fro');
                        
            % Compute the reconstruction error between the submatrices
            err_in_D_X(1, trial) = norm(edm4_D_X2 - D_X2, 'fro');
            rel_err_in_D_X(1,trial) = norm(edm4_D_X2 - D_X2, 'fro')/ norm(D_X2, 'fro');

        
        elseif strcmp(completion_method, 'sdr')
            % Applying Semidefinite Relaxation 
            edm6 = sdr_complete_edm(Dtot_masked, mask, d);
            edm6_D_X2 = edm6(1:n, 1:n);

    
            % Calculate error for Soft-Impute
            err_in(1, trial) = norm(edm6 - Dtot, 'fro');
            rel_err_in(1, trial) = norm(edm6 - Dtot, 'fro') / norm(Dtot, 'fro');
                        
            % Compute the reconstruction error between the submatrices
            err_in_D_X(1, trial) = norm(edm6_D_X2 - D_X2, 'fro');
            rel_err_in_D_X(1,trial) = norm(edm6_D_X2 - D_X2, 'fro')/ norm(D_X2, 'fro');

        end
    end

    % Compute the average error and standard deviation over the trials
    err = mean(err_in, 2);
    err_sd = std(err_in, 0, 2); % Use '0' as the second argument for normalization
    rel_err = mean(rel_err_in, 2);
    rel_err_sd= std(rel_err_in, 0, 2); % Use '0' as the second argument for normalization

    err_DX2 = mean(err_in_D_X, 2);
    err_DX2_sd = std(err_in_D_X, 0, 2); % Use '0' as the second argument for normalization
    rel_err_DX2 = mean(rel_err_in_D_X, 2);
    rel_err_DX2_sd= std(rel_err_in_D_X, 0, 2); % Use '0' as the second argument for normalization


    
end


function [err_mc, err_sd_mc, rel_err_mc, rel_err_sd_mc, err_DX2_mc, err_DX2_sd_mc, rel_err_DX2_mc, rel_err_DX2_sd_mc] = run_monte_carlo_simulation(X, nys_array, num_trials, nys_min, nys_max, n, d, nyst_method, completion_method, folder, dataset_name)

    num_nys = numel(nys_array);


    % Preallocate result arrays
    err_mc = zeros(1, num_nys);
    err_sd_mc = zeros(1, num_nys);
    rel_err_mc = zeros(1, num_nys);
    rel_err_sd_mc = zeros(1, num_nys);

    err_DX2_mc = zeros(1, num_nys);
    err_DX2_sd_mc = zeros(1, num_nys);
    rel_err_DX2_mc = zeros(1, num_nys);
    rel_err_DX2_sd_mc = zeros(1, num_nys);


    % Create a data queue
    dq = parallel.pool.DataQueue();
    
    % Define the afterEach function to process the results and save to files
    afterEach(dq, @(data) saveTrialResults(data, folder));


    parfor j_nys = 1:num_nys
        fprintf('#(Nystrom) %d in the range %d-%d\n', nys_array(j_nys), nys_min, nys_max);
        t0 = tic;
        Mnys = nys_array(j_nys);
    
        % Execute for a specific number of Nystrom points
        [err_mc_iter, err_sd_mc_iter, rel_err_mc_iter, rel_err_mc_sd_iter, err_DX2_mc_iter,err_DX2_sd_mc_iter, rel_err_DX2_mc_iter,rel_err_DX2_sd_mc_iter] = run_single_nystrom_trial(X, Mnys, num_trials, n, d, nyst_method, completion_method);

        % Enqueue the results for later processing
        send(dq, struct('Mnys', Mnys, 'err_mc_iter', err_mc_iter, 'err_sd_mc_iter', err_sd_mc_iter, 'rel_err_mc_iter', rel_err_mc_iter, 'rel_err_mc_sd_iter', rel_err_mc_sd_iter, 'err_DX2_mc_iter', err_DX2_mc_iter,'err_DX2_sd_mc_iter', err_DX2_sd_mc_iter, 'rel_err_DX2_mc_iter', rel_err_DX2_mc_iter,'rel_err_DX2_sd_mc_iter', rel_err_DX2_sd_mc_iter));

        elapsed_time = toc(t0);
        fprintf('%d trials for Nystrom %d time: %.2f seconds\n', num_trials, Mnys, elapsed_time);

        % Store the results of all trials for this Nystrom point count in the overall results arrays
        err_mc(:, j_nys) = err_mc_iter;
        err_sd_mc(:, j_nys) = err_sd_mc_iter;
        rel_err_mc(:, j_nys) = rel_err_mc_iter;
        rel_err_sd_mc(:, j_nys) = rel_err_mc_sd_iter;

        err_DX2_mc(:, j_nys) =  err_DX2_mc_iter;
        err_DX2_sd_mc(:, j_nys) =  err_DX2_sd_mc_iter;
        rel_err_DX2_mc(:, j_nys) = rel_err_DX2_mc_iter;
        rel_err_DX2_sd_mc(:, j_nys) = rel_err_DX2_sd_mc_iter
    end
    % Signal the end of data
    send(dq, []);

end

    

% Define the function to process the data and save to files
function saveTrialResults(data, folder)
    if ~isempty(data)
        Mnys = data.Mnys;
        err_mc = data.err_mc_iter;
        err_sd_mc = data.err_sd_mc_iter;
        rel_err_mc = data.rel_err_mc_iter;
        rel_err_mc_sd = data.rel_err_mc_sd_iter;

        err_DX2_mc = data.err_DX2_mc_iter;
        err_DX2_sd_mc = data.err_DX2_sd_mc_iter;
        rel_err_DX2_mc = data.rel_err_DX2_mc_iter;
        rel_err_DX2_sd_mc = data.rel_err_DX2_sd_mc_iter;





        % Save results for this trial
        subfolder = sprintf('%s/Nystrom%d', folder, Mnys);  % Create subfolder for each Nystrom point
        if ~isfolder(subfolder)
            mkdir(subfolder);
        end

        matrix_error = sprintf('%s/ErrorVals%d.mat', subfolder, Mnys);
        matrix_err_std = sprintf('%s/ErrorStd%d.mat', subfolder, Mnys);
        matrix_rel_error = sprintf('%s/Rel_ErrorVals%d.mat', subfolder, Mnys);
        matrix_rel_err_std = sprintf('%s/Rel_ErrorStd%d.mat', subfolder, Mnys);

        matrix_err_DX2 = sprintf('%s/ErrorValsInDX2%d.mat', subfolder, Mnys);
        matrix_err_DX2_std = sprintf('%s/ErrorStdInDX2%d.mat', subfolder, Mnys);
        matrix_rel_error_DX2 = sprintf('%s/Rel_ErrorValsInDX2%d.mat', subfolder, Mnys);
        matrix_rel_err_DX2_std = sprintf('%s/Rel_ErrorStdInDX2%d.mat', subfolder, Mnys);

        save(matrix_error, 'err_mc');
        save(matrix_err_std, 'err_sd_mc');
        save(matrix_rel_error, 'rel_err_mc');
        save(matrix_rel_err_std, 'rel_err_mc_sd');

        save(matrix_err_DX2, 'err_DX2_mc');
        save(matrix_err_DX2_std, 'err_DX2_sd_mc');
        save(matrix_rel_error_DX2, 'rel_err_DX2_mc');
        save(matrix_rel_err_DX2_std, 'rel_err_DX2_sd_mc');
    end
end
