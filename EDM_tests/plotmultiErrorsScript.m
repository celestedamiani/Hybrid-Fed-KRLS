clear all; 
plotmultiErrors('Sonar', 200, 'Sonar', {'Alternating_Descent', 'Rank_Alternation', 'Soft-Impute'}, 'subsampling');

%{'sdr', 'Alternating_Descent', 'OptSpace', 'Rank_Alternation', 'Soft-Impute'}

function plotmultiErrors(parentFolderPath, nys_max, dataset_name, completion_methods, nyst_method)
    legend_labels = strrep(completion_methods, '_', ' ');
    % Initialize arrays to store data for all methods
    num_methods = numel(completion_methods);
    nys_array = 10:10:nys_max;
    err_mc_all_methods = NaN(numel(nys_array), num_methods);
    std_mc_all_methods = NaN(numel(nys_array), num_methods);
    rel_err_mc_all_methods = NaN(numel(nys_array), num_methods);
    rel_err_std_mc_all_methods = NaN(numel(nys_array), num_methods);
    err_DX2_mc_all_methods = NaN(numel(nys_array), num_methods);
    std_DX2_mc_all_methods = NaN(numel(nys_array), num_methods);
    rel_err_DX2_mc_all_methods = NaN(numel(nys_array), num_methods);
    rel_err_std_DX2_mc_all_methods = NaN(numel(nys_array), num_methods);


    % Loop through each completion method
    for method_idx = 1:num_methods
        completion_method = completion_methods{method_idx};

        % Loop through each Nystrom folder for the current method
        for i = 1:numel(nys_array)
            nys_folder = fullfile(parentFolderPath, completion_method, nyst_method, sprintf('Nystrom%d', nys_array(i)));
            disp(nys_folder);
            err_file = fullfile(nys_folder, sprintf('ErrorVals%d.mat', nys_array(i)));
            err_std_file = fullfile(nys_folder, sprintf('ErrorStd%d.mat', nys_array(i)));
            rel_err_file = fullfile(nys_folder, sprintf('Rel_ErrorVals%d.mat', nys_array(i)));
            rel_err_std_file = fullfile(nys_folder, sprintf('Rel_ErrorStd%d.mat', nys_array(i)));
            
            err_DX2_file = fullfile(nys_folder, sprintf('ErrorValsInDX2%d.mat', nys_array(i)));
            err_std_DX2_file = fullfile(nys_folder, sprintf('ErrorStdInDX2%d.mat', nys_array(i)));
            rel_err_DX2_file = fullfile(nys_folder, sprintf('Rel_ErrorValsInDX2%d.mat', nys_array(i)));
            rel_err_std_DX2_file = fullfile(nys_folder, sprintf('Rel_ErrorStdInDX2%d.mat', nys_array(i)));

            if exist(err_file, 'file') == 2 && exist(err_std_file, 'file') == 2 && exist(rel_err_file, 'file') == 2 && exist(rel_err_std_file, 'file') == 2 && ...
                    exist(err_DX2_file, 'file') == 2  && exist(err_std_DX2_file, 'file') == 2  && exist(rel_err_DX2_file, 'file') == 2  && exist(rel_err_std_DX2_file, 'file') == 2 
                % Load error data
                load(err_file, 'err_mc');
                load(err_std_file, 'err_sd_mc');
                err_mc_all_methods(i, method_idx) = err_mc;
                std_mc_all_methods(i, method_idx) = err_sd_mc;

                % Load relative error data
                load(rel_err_file, 'rel_err_mc');
                load(rel_err_std_file, 'rel_err_mc_sd');
                rel_err_mc_all_methods(i, method_idx) = rel_err_mc;
                rel_err_std_mc_all_methods(i, method_idx) = rel_err_mc_sd;
                % Convert relative error to percentage after loading the data
                rel_err_mc_all_methods_percent = rel_err_mc_all_methods * 100;
                rel_err_std_mc_all_methods_percent = rel_err_std_mc_all_methods * 100;

              
                % Load error data in DX.^2
                load(err_DX2_file, 'err_DX2_mc');
                load(err_std_DX2_file, 'err_DX2_sd_mc');
                err_mc_DX2_all_methods(i, method_idx) = err_DX2_mc;
                std_mc_DX2_all_methods(i, method_idx) = err_DX2_sd_mc;

                % Load relative error data in DX.^2
                load(rel_err_DX2_file, 'rel_err_DX2_mc');
                load(rel_err_std_DX2_file, 'rel_err_DX2_sd_mc');
                rel_err_mc_DX2_all_methods(i, method_idx) = rel_err_DX2_mc;
                rel_err_std_mc_DX2_all_methods(i, method_idx) = rel_err_DX2_sd_mc;
                % Convert relative error to percentage after loading the data
                rel_err_mc_DX2_all_methods_percent = rel_err_mc_DX2_all_methods * 100;
                rel_err_std_mc_DX2_all_methods_percent = rel_err_std_mc_DX2_all_methods * 100;




            else
                warning('Files not found for method %s, Nystrom %d', completion_method, nys_array(i));
            end
        end
    end

    % Plotting absolute error for all methods
    figure;
    hold on;
    line_styles = {'-', '--', ':', '-.'}; % Add more line styles if needed
    marker_types = {'o', 's', '^', 'd'}; % Add more marker types if needed
    colors = lines(num_methods); 
    
    for method_idx = 1:num_methods
        % Customize line style, marker type, and color for each method
        errorbar(nys_array, err_mc_all_methods(:, method_idx), std_mc_all_methods(:, method_idx),...
            'LineStyle', line_styles{method_idx}, 'LineWidth', 1.5, 'Marker', marker_types{method_idx},...
            'Color', colors(method_idx,:), 'MarkerSize', 5);
    end

    ylabel('Error');
    xlabel('Number of Nystrom points');
    grid on;
    title(sprintf('%s dataset: Absolute Error for %s', dataset_name, nyst_method));
    legend(legend_labels, 'Location', 'northeast')
    hold off;
    % Set y-axis limit to start from 0
    ylim([0, max(max(err_mc_all_methods + std_mc_all_methods))]); 
    saveas(gcf, fullfile(parentFolderPath, sprintf('AbsoluteError_%s_%s.png', dataset_name, nyst_method)));

    % Plotting relative error for all methods
    figure;
    hold on;
    line_styles = {'-', '--', ':', '-.'}; % Add more line styles if needed
    marker_types = {'o', 's', '^', 'd'}; % Add more marker types if needed
    colors = lines(num_methods); 
    
    for method_idx = 1:num_methods
        % Customize line style, marker type, and color for each method
        errorbar(nys_array, rel_err_mc_all_methods_percent(:, method_idx), rel_err_std_mc_all_methods_percent(:, method_idx),...
            'LineStyle', line_styles{method_idx}, 'LineWidth', 1.5, 'Marker', marker_types{method_idx},...
            'Color', colors(method_idx,:), 'MarkerSize', 5);
    end

    ylabel('Relative error (%)');
    xlabel('Number of Nystrom points');
    grid on;
    title(sprintf('%s dataset: Relative Error for %s', dataset_name, nyst_method));
    legend(legend_labels, 'Location', 'northeast')
    hold off;
    ylim([0, max(max(rel_err_mc_all_methods_percent + rel_err_std_mc_all_methods_percent))]); 
    saveas(gcf, fullfile(parentFolderPath, sprintf('RelativeErrorPercent_%s_%s.png', dataset_name, nyst_method)));

    % Plotting absolute error for all methods IN D_X.^2
    figure;
    hold on;
    line_styles = {'-', '--', ':', '-.'}; % Add more line styles if needed
    marker_types = {'o', 's', '^', 'd'}; % Add more marker types if needed
    colors = lines(num_methods); 
    
    for method_idx = 1:num_methods
        % Customize line style, marker type, and color for each method
        errorbar(nys_array, err_mc_DX2_all_methods(:, method_idx), std_mc_DX2_all_methods(:, method_idx),...
            'LineStyle', line_styles{method_idx}, 'LineWidth', 1.5, 'Marker', marker_types{method_idx},...
            'Color', colors(method_idx,:), 'MarkerSize', 5);
    end

    ylabel('Error');
    xlabel('Number of Nystrom points');
    grid on;
    title(sprintf('%s dataset: Absolute Error for %s in DX^2', dataset_name, nyst_method));
    legend(legend_labels, 'Location', 'northeast')
    hold off;
    ylim([0, max(max(err_mc_DX2_all_methods + std_mc_DX2_all_methods))]); 
    saveas(gcf, fullfile(parentFolderPath, sprintf('AbsoluteErrorDX2_%s_%s.png', dataset_name, nyst_method)));

    % Plotting relative error for all methods IN D_X.^2
    figure;
    hold on;
    line_styles = {'-', '--', ':', '-.'}; % Add more line styles if needed
    marker_types = {'o', 's', '^', 'd'}; % Add more marker types if needed
    colors = lines(num_methods); 
    
    for method_idx = 1:num_methods
        % Customize line style, marker type, and color for each method
        errorbar(nys_array, rel_err_mc_DX2_all_methods_percent(:, method_idx), rel_err_std_mc_DX2_all_methods_percent(:, method_idx),...
            'LineStyle', line_styles{method_idx}, 'LineWidth', 1.5, 'Marker', marker_types{method_idx},...
            'Color', colors(method_idx,:), 'MarkerSize', 5);
    end

    ylabel('Relative error (%)');
    xlabel('Number of Nystrom points');
    grid on;
    title(sprintf('%s dataset: Relative Error for %s in DX^2', dataset_name, nyst_method));
    legend(legend_labels, 'Location', 'northeast')
    hold off;
    ylim([0, max(max(rel_err_mc_DX2_all_methods_percent + rel_err_std_mc_DX2_all_methods_percent))]); 

    saveas(gcf, fullfile(parentFolderPath, sprintf('RelativeErrorPercentDX2_%s_%s.png', dataset_name, nyst_method)));

end
