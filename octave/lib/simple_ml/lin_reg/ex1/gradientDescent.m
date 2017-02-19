function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    %GRADIENTDESCENT Performs gradient descent to learn theta
    %   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    %   taking num_iters gradient steps with learning rate alpha

    % Initialize some useful values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters

        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta.
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %

        h_x = X * theta;

        % % Update for theta_0
        % d_J0 = (h_x - y)' * X(:, 1)
        % update_0 = alpha * (1 / m) * sum(d_J0)
        % theta(1) = theta(1) - update_0

        % % Update for theta_1
        % d_J1 = (h_x - y)' * X(:, 2)
        % update_1 = alpha * (1 / m) * sum(d_J1)
        % theta(2) = theta(2) - update_1

        d_J = (h_x - y)' * X;
        update = alpha * (1 / m) * d_J;
        theta = theta - update';

        % ============================================================

        fprintf('Cost: %f\n', computeCost(X, y, theta));
        % Save the cost J in every iteration
        J_history(iter) = computeCost(X, y, theta);

    end

end
