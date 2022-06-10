% gamma = 0.9 / epsilon function : 200 / (200 + k); please load 'eval.m'
% It takes about 10 sec to get the result.
gamma = 0.9;
[reach_goal, exe_time, qevalstates] = Qlearning(qevalreward, gamma, 5);
display(qevalstates)
function [reach_goal, exe_time, state_map] = Qlearning(reward, gamma_value, eps_type)
trial_max = 3000;
exe_time = 0;
reach_goal = 0;
run_times = 10;
q_threshold = 0.05;
max_reward = 0;
opt_policy = [];
for r = 1 : run_times
    Q = zeros(size(reward));
    tic;
    for trial = 1 : trial_max
        s_k = 1; k = 1;
        Q_pre = Q;
        while s_k ~= 100
            eps_k = getEps(k, eps_type);
            if eps_k < 0.005
                break;
            end
            % next action
            a_k = nextAct(Q(s_k, :), eps_k, reward(s_k, :));
            % next state
            nxt = 10 ^ (mod(a_k + 1, 2)) * (-1) ^ (floor(a_k / 2) + 1);
            % update Q value
            Q(s_k, a_k) = Q(s_k, a_k) + eps_k * (reward(s_k, a_k) + gamma_value * max(Q(s_k + nxt, :)) - Q(s_k, a_k));
            s_k = s_k + nxt;
            k = k + 1;
        end
        if max(abs(Q_pre - Q)) < q_threshold
            break;
        end
    end
    [total_reward, cur_path] = optPolicay(Q, gamma_value, reward);
    % judge whether reach target
    if cur_path(length(cur_path)) == 100
        reach_goal = reach_goal + 1;
        exe_time = exe_time + toc;
        if total_reward > max_reward
            max_reward = total_reward;
            opt_policy = cur_path;
        end
    end  
end
if reach_goal ~= 0
    exe_time = exe_time / reach_goal;
end
if max_reward ~= 0
    disp(['When gamma = ', num2str(gamma_value),' epsilon type=', num2str(eps_type), ' total_reward: ', num2str(total_reward)])
    drawOptPol(Q, max_reward)
    drawTraj(max_reward, opt_policy)
end
opt_policy = opt_policy(1:end-1)';
state_map = ones(length(opt_policy)+1, 1);
for i = 1 : length(opt_policy)
    a_k = opt_policy(i);
    state_map(i + 1) = state_map(i) + 10 ^ (mod(a_k + 1, 2)) * (-1) ^ (floor(a_k / 2) + 1);
end
end

% choose epsilon
function [eps_k] = getEps(k, eps_mode)
if eps_mode == 1
    eps_k = 1 / k;
elseif eps_mode == 2
    eps_k = 100 / (100 + k);
elseif eps_mode == 3
    eps_k = (1 + log(k)) / k;
elseif eps_mode == 4
    eps_k = (1 + 5 * log(k)) / k;
else
    eps_k = 200 / (200 + k);
end
end

% choose next action
function act_nxt = nextAct(Q_sk, eps_k, reward_sk)
valid_idx=find(reward_sk ~= -1);
if sum(Q_sk) ~= 0
    if rand > eps_k
        [~, idx] = max(Q_sk(valid_idx));
        act_nxt = valid_idx(idx);
    else
        rand_idx = find(Q_sk(valid_idx) ~= max(Q_sk(valid_idx)));
        idx = randperm(length(rand_idx), 1);
        act_nxt = valid_idx(rand_idx(idx));
    end
else
    idx = randperm(length(valid_idx), 1);
    act_nxt = valid_idx(idx);
end
end

% obtain optimal policy
function [total_reward, cur_path] = optPolicay(Q, gamma_val, reward)
[~, actions]=max(Q,[],2);
s = 1;
discount = 1;
dp = zeros(1, 100);
cur_path = [];
total_reward = 0;
while s ~= 100 && dp(s) == 0
    cur_path = [cur_path actions(s)];
    dp(s) = discount * reward(s, actions(s));
    total_reward = total_reward + dp(s);
    s = s + (10 ^ (mod(actions(s) + 1, 2)) * (-1) ^ (floor(actions(s) / 2) + 1));
    discount = discount * gamma_val;
end
if s == 100
    cur_path = [cur_path s];
end
end

% draw execution of policy
function [] = drawTraj(total_reward, path_opt)
n = length(path_opt);
pos = 1;
direction = ['^b';'>b';'vb';'<b'];
figure
hold on
plot(9.5, 9.5, '*r')
axis([0 10 0 10])
title(['Execution of optimal policy with associated reward = ', num2str(total_reward)])
grid on
set(gca,'YDir','reverse')
for i = 1 : n - 1
    x = floor((pos  - 1) / 10) + 0.5;
    y = mod(pos - 1, 10) + 0.5;
    a_k = path_opt(i);
    plot(x, y, direction(a_k, :))
    nxt = 10 ^ (mod(a_k + 1, 2))  * (-1) ^ (floor(a_k / 2) + 1);
    pos = pos + nxt;
end
hold off
end

% draw optimal policy
function [] = drawOptPol(Q, total_reward)
[~, opt_act]=max(Q,[],2);
n = length(opt_act);
direction = ['^r';'>r';'vr';'<r'];
figure
hold on
plot(9.5, 9.5, '*b')
axis([0 10 0 10])
title(['Optimal policy with associated reward = ', num2str(total_reward)])
grid on
set(gca,'YDir','reverse')
for i = 1 : n
    x = floor((i  - 1) / 10) + 0.5;
    y = mod(i - 1, 10) + 0.5;
    plot(x, y, direction(opt_act(i), :))
end
hold off
end