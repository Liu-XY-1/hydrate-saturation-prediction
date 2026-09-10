warning off
close all
clear
clc

codeDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(codeDir);
xgboostDir = fullfile(projectDir, 'third_party', 'xgboost');

dllPath = fullfile(xgboostDir, 'xgboost.dll');
hdrPath = fullfile(xgboostDir, 'xgboost.h');

addpath(xgboostDir);

try
    unloadlibrary('xgboost');
catch
end

try
    loadlibrary(dllPath, hdrPath, 'alias', 'xgboost');
    disp('XGBoost library loaded successfully');
catch ME
    error('XGBoost library load failed: %s\n\nPlease make sure xgboost.dll and xgboost.h exist in the third_party/xgboost folder.', ...
        ME.message);
end


codeDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(codeDir);

dataDir = fullfile(projectDir, 'data');
res = xlsread(fullfile(dataDir, 'W17_public.xlsx'));


load(fullfile(dataDir, 'train.mat'), ...
     'trainIdx', 'valIdx', 'testIdx');

Depth_train = res(trainIdx, 1);
Depth_val   = res(valIdx,   1);
Depth_test  = res(testIdx,  1);

X_train_raw = res(trainIdx, 2:7);
Y_train_raw = res(trainIdx, 8);
X_val_raw   = res(valIdx,   2:7);
Y_val_raw   = res(valIdx,   8);
X_test_raw  = res(testIdx,  2:7);
Y_test_raw  = res(testIdx,  8);

M = size(X_train_raw, 1);
V = size(X_val_raw,   1);
N = size(X_test_raw,  1);

[p_train, ps_input] = mapminmax(X_train_raw', 0, 1);
[t_train, ps_output] = mapminmax(Y_train_raw', 0, 1);

p_val = mapminmax('apply', X_val_raw',  ps_input);
t_val = mapminmax('apply', Y_val_raw',  ps_output);

p_test  = mapminmax('apply', X_test_raw', ps_input);
t_test  = mapminmax('apply', Y_test_raw', ps_output);

p_train = p_train';
p_val   = p_val';
p_test  = p_test';
t_train = t_train';
t_val   = t_val';
t_test  = t_test';

n_particles = 45;
max_iter = 80;
w = 0.45;
c1 = 2;
c2 = 2;

var_min = [0.05, 8, 150, 0, 0, 0.70, 0.8];
var_max = [0.20, 18, 600, 4, 2, 1.0, 1.0];

particle_position = rand(n_particles, 7) .* (var_max - var_min) + var_min;
particle_velocity = zeros(n_particles, 7);
particle_best_position = particle_position;
particle_best_fitness = inf(n_particles, 1);

global_best_position = zeros(1, 7);
global_best_fitness = inf;

for iter = 1:max_iter
    for i = 1:n_particles
        eta = particle_position(i, 1);
        max_depth = round(particle_position(i, 2));
        num_trees = round(particle_position(i, 3));
        lambda = particle_position(i, 4);
        alpha = particle_position(i, 5);
        subsample = particle_position(i, 6);
        colsample_bytree = particle_position(i, 7);

        params.eta = eta;
        params.objective = 'reg:squarederror';
        params.max_depth = max_depth;
        params.lambda = lambda;
        params.alpha = alpha;
        params.subsample = subsample;
        params.colsample_bytree = colsample_bytree;
        params.min_child_weight = 5;

        model = xgboost_train(p_train, t_train, params, num_trees);

        v_sim = xgboost_test(p_val, model);
        V_sim = mapminmax('reverse', v_sim', ps_output)';
        fitness = sqrt(sum((V_sim - Y_val_raw).^2) ./ V);

        if fitness < particle_best_fitness(i)
            particle_best_fitness(i) = fitness;
            particle_best_position(i, :) = particle_position(i, :);
            if fitness < global_best_fitness
                global_best_fitness = fitness;
                global_best_position = particle_position(i, :);
            end
        end
    end

    for i = 1:n_particles
        particle_velocity(i, :) = w * particle_velocity(i, :) ...
            + c1 * rand(1, 7) .* (particle_best_position(i, :) - particle_position(i, :)) ...
            + c2 * rand(1, 7) .* (global_best_position - particle_position(i, :));
        particle_position(i, :) = particle_position(i, :) + particle_velocity(i, :);
        particle_position(i, :) = max(particle_position(i, :), var_min);
        particle_position(i, :) = min(particle_position(i, :), var_max);
    end

    disp(['PSO Iteration ' num2str(iter) ': Best Val RMSE = ' num2str(global_best_fitness)]);
end

best_eta = global_best_position(1);
best_max_depth = round(global_best_position(2));
best_num_trees = round(global_best_position(3));
best_lambda = global_best_position(4);
best_alpha = global_best_position(5);
best_subsample = global_best_position(6);
best_colsample_bytree = global_best_position(7);

disp(['Best eta: ' num2str(best_eta)]);
disp(['Best max_depth: ' num2str(best_max_depth)]);
disp(['Best num_trees: ' num2str(best_num_trees)]);
disp(['Best lambda: ' num2str(best_lambda)]);
disp(['Best alpha: ' num2str(best_alpha)]);
disp(['Best subsample: ' num2str(best_subsample)]);
disp(['Best colsample_bytree: ' num2str(best_colsample_bytree)]);

params.eta = best_eta;
params.objective = 'reg:squarederror';
params.max_depth = best_max_depth;
params.lambda = best_lambda;
params.alpha = best_alpha;
params.subsample = best_subsample;
params.colsample_bytree = best_colsample_bytree;
params.min_child_weight = 5;
num_trees = best_num_trees;

model = xgboost_train(p_train, t_train, params, num_trees);

t_sim1 = xgboost_test(p_train, model);
t_simV = xgboost_test(p_val,   model);
t_sim2 = xgboost_test(p_test,  model);

T_sim1 = mapminmax('reverse', t_sim1', ps_output)';
T_simV = mapminmax('reverse', t_simV', ps_output)';
T_sim2 = mapminmax('reverse', t_sim2', ps_output)';

[Depth_train_sorted, idx_train] = sort(Depth_train);
T_train_sorted = Y_train_raw(idx_train);
T_sim1_sorted  = T_sim1(idx_train);

[Depth_val_sorted, idx_val] = sort(Depth_val);
T_val_sorted  = Y_val_raw(idx_val);
T_simV_sorted = T_simV(idx_val);

[Depth_test_sorted, idx_test] = sort(Depth_test);
T_test_sorted = Y_test_raw(idx_test);
T_sim2_sorted = T_sim2(idx_test);

error1 = sqrt(sum((T_sim1 - Y_train_raw).^2) ./ M);
errorV = sqrt(sum((T_simV - Y_val_raw).^2) ./ V);
error2 = sqrt(sum((T_sim2 - Y_test_raw).^2) ./ N);

disp(['Training subset RMSE: ', num2str(error1)])
disp(['Validation set RMSE: ', num2str(errorV)])
disp(['Test set RMSE: ', num2str(error2)])

R1 = 1 - norm(Y_train_raw - T_sim1)^2 / norm(Y_train_raw - mean(Y_train_raw))^2;
RV = 1 - norm(Y_val_raw  - T_simV)^2 / norm(Y_val_raw  - mean(Y_val_raw))^2;
R2 = 1 - norm(Y_test_raw  - T_sim2)^2 / norm(Y_test_raw  - mean(Y_test_raw))^2;

disp(['Training subset R2: ', num2str(R1)])
disp(['Validation set R2: ', num2str(RV)])
disp(['Test set R2: ', num2str(R2)])

mae1 = sum(abs(T_sim1 - Y_train_raw)) ./ M;
maeV = sum(abs(T_simV - Y_val_raw)) ./ V;
mae2 = sum(abs(T_sim2 - Y_test_raw)) ./ N;

mbe1 = sum(T_sim1 - Y_train_raw) ./ M;
mbeV = sum(T_simV - Y_val_raw) ./ V;
mbe2 = sum(T_sim2 - Y_test_raw) ./ N;

disp(['Training subset MAE: ', num2str(mae1)])
disp(['Validation set MAE: ', num2str(maeV)])
disp(['Test set MAE: ', num2str(mae2)])
disp(['Training subset MBE: ', num2str(mbe1)])
disp(['Validation set MBE: ', num2str(mbeV)])
disp(['Test set MBE: ', num2str(mbe2)])

figure
plot(Depth_train_sorted, T_train_sorted, 'r-', Depth_train_sorted, T_sim1_sorted, 'b-', 'LineWidth', 1)
legend('True','Prediction')
xlabel('Depth (m)')
ylabel('Saturation')
string = {'Training subset prediction comparison'; ['RMSE=' num2str(error1) ', R^2=' num2str(R1)]};
title(string)
grid

figure
plot(Depth_val_sorted, T_val_sorted, 'r-', Depth_val_sorted, T_simV_sorted, 'b-', 'LineWidth', 1)
legend('True','Prediction')
xlabel('Depth (m)')
ylabel('Saturation')
string = {'Validation set prediction comparison'; ['RMSE=' num2str(errorV) ', R^2=' num2str(RV)]};
title(string)
grid

figure
plot(Depth_test_sorted, T_test_sorted, 'r-', Depth_test_sorted, T_sim2_sorted, 'b-', 'LineWidth', 1)
legend('True', 'Prediction')
xlabel('Depth (m)')
ylabel('Saturation')
string = {'Test set prediction comparison'; ['RMSE=' num2str(error2) ', R^2=' num2str(R2)]};
title(string)
grid

sz = 25; c = 'b';
figure
scatter(Y_train_raw, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('Training subset true values'); ylabel('Training subset predictions');
xlim([min(Y_train_raw) max(Y_train_raw)])
ylim([min(T_sim1) max(T_sim1)])
title('Training subset predictions vs. true values')

figure
scatter(Y_test_raw, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('Test set true values'); ylabel('Test set predictions');
xlim([min(Y_test_raw) max(Y_test_raw)])
ylim([min(T_sim2) max(T_sim2)])
title('Test set predictions vs. true values')

function model = xgboost_train(p_train, t_train, params, max_num_iters)

loadlibrary('xgboost', ...
    fullfile(fileparts(mfilename('fullpath')), '..', 'third_party', 'xgboost', 'xgboost.h'))

missing = single(NaN);
iters_optimal = max_num_iters;

if isempty(params)
    params.booster           = 'gbtree';
    params.objective         = 'reg:linear';
    params.max_depth         = 5;
    params.eta               = 0.1;
    params.min_child_weight  = 1;
    params.subsample         = 0.9;
    params.colsample_bytree  = 1;
    params.num_parallel_tree = 1;
else
    if ~isfield(params, 'booster')
        params.booster = 'gbtree';
    end
    if ~isfield(params, 'min_child_weight')
        params.min_child_weight = 1;
    end
    if ~isfield(params, 'num_parallel_tree')
        params.num_parallel_tree = 1;
    end
end

param_fields = fields(params);
for i = 1 : length(param_fields)
    eval(['params.' param_fields{i} ' = num2str(params.' param_fields{i} ');'])
end

rows = uint64(size(p_train, 1));
cols = uint64(size(p_train, 2));
p_train = p_train';

p_train_ptr = libpointer('singlePtr', single(p_train));
t_train_ptr = libpointer('singlePtr', single(t_train));

h_train_ptr = libpointer;
h_train_ptr_ptr = libpointer('voidPtrPtr', h_train_ptr);

calllib('xgboost', 'XGDMatrixCreateFromMat', p_train_ptr, rows, cols, missing, h_train_ptr_ptr);

labelStr = 'label';
calllib('xgboost', 'XGDMatrixSetFloatInfo', h_train_ptr, labelStr, t_train_ptr, rows);

h_booster_ptr = libpointer;
h_booster_ptr_ptr = libpointer('voidPtrPtr', h_booster_ptr);
calllib('xgboost', 'XGBoosterCreate', h_train_ptr_ptr, uint64(1), h_booster_ptr_ptr);

for i = 1 : length(param_fields)
    eval(['calllib(''xgboost'', ''XGBoosterSetParam'', h_booster_ptr, ''' param_fields{i} ''', ''' eval(['params.' param_fields{i}]) ''');'])
end

for iter = 1 : iters_optimal
    calllib('xgboost', 'XGBoosterUpdateOneIter', h_booster_ptr, int32(iter), h_train_ptr);
end

model                = struct;
model.iters_optimal  = iters_optimal;
model.h_booster_ptr  = h_booster_ptr;
model.params         = params;
model.missing        = missing;

end

function Yhat = xgboost_test(p_test, model)

h_booster_ptr = model.h_booster_ptr;

rows = uint64(size(p_test, 1));
cols = uint64(size(p_test, 2));
p_test = p_test';

h_test_ptr = libpointer;
h_test_ptr_ptr = libpointer('voidPtrPtr', h_test_ptr);
test_ptr = libpointer('singlePtr', single(p_test));
calllib('xgboost', 'XGDMatrixCreateFromMat', test_ptr, rows, cols, model.missing, h_test_ptr_ptr);

out_len_ptr = libpointer('uint64Ptr', uint64(0));
f = libpointer('singlePtr');
f_ptr = libpointer('singlePtrPtr', f);
calllib('xgboost', 'XGBoosterPredict', h_booster_ptr, h_test_ptr, int32(0), uint32(0), int32(0), out_len_ptr, f_ptr);

n_outputs = out_len_ptr.Value;
setdatatype(f, 'singlePtr', n_outputs);

Yhat = double(f.Value);

end
