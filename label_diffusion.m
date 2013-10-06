function predictions_score = label_diffusion(network, labels, known_nodes, nodes_to_predict, options)
% Label diffusion method for semi-supervised graph labelisation.
% Based on the method described in D. Zhou et al, Learning with Local and Global
% Consistency. (adapted to graphs)
%
% INPUT:
% - network : adjacency matrix of the network.
% - labels : [m x k] matrix. m is the number of nodes whose labels are
%   known. k is the number of labels. Element ij of labels is 1 if ith
%   known node has label j. 0 else.
% - known_nodes : vector of size m. contains indices of nodes whose labels are known.
% - nodes_to_predict : vector that contains indices of nodes whose
%   labels must be predicted
% - options
%   - options.alpha : propagation parameter between 0 and 1. (default : 0.99)
%   - options.precision : level of precision required before stoping
%     iterations. (default : 1e-8)
%   - options.maxiter : maximum number of iterations (default : 100)
%
% OUTPUT: 
% "predictions_score", matrix whose element ij is the predicted score of
% label j for the ith node to predict.
%
% Robin Devooght 2013, october 6th

% Gives default values to unspecified options
if ~isfield(options, 'alpha')
    options.alpha = 0.99;
end
if ~isfield(options, 'precision')
    options.precision = 1e-5;
end
if ~isfield(options, 'maxiter')
    options.maxiter = 100;
end

[n, m] = size(network);

if n ~= m
    error('label_diffusion:A_square', 'Adjacency matrix must be square');
end
if options.alpha <= 0 || options.alpha >= 1
    error('label_diffusion:alpha_range', 'alpha must belong to ]0,1[');
end

% Normalization of the adjacency matrix
D = diag(sum(network,2).^(-0.5));
diffusion_matrix = D*network*D;

k = size(labels, 2);
ini_scores = zeros(n, k);
ini_scores(known_nodes, :) = labels;
predictions_score = ini_scores;

% Propagation
for i=1:options.maxiter
    last_score = predictions_score;
    predictions_score = options.alpha*diffusion_matrix*predictions_score + (1-options.alpha)*ini_scores;
    if max(max(abs(last_score-predictions_score))) < options.precision
        break;
    end
end

% keep only predictions for nodes specified by 'nodes_to_predict'
predictions_score = predictions_score(nodes_to_predict, :);