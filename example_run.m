% Demonstration of the use of label_diffusion.
% Based on "maindriver.m" by Lei Tang (see http://leitang.net/code/social-dimension/SocioDim.zip)
% Robin Devooght 2013, october 6th

load blogcatalog.mat

% randomly generate index_tr (training nodes index) and index_te (test nodes index)
n = size(network, 1);
index = randperm(n);
index_tr = index(1:ceil(0.1*n));  % 10% labeled nodes for training
index_te = index(1+ceil(0.1*n):end);  % 90% unlabeled nodes for test
labels = group(index_tr, :); % the labels of nodes for training

% Diffusion process
options.alpha = .5; % Atenuation of the diffusion
[predscore] = label_diffusion(network, labels, index_tr, index_te, options);

[perf, pred] = evaluate(predscore, group(index_te, :));

perf.micro_F1
perf.macro_F1
perf.acc