function average_error = grad_check(fun, theta0, num_checks, varargin)

  % This function implements gradient checking by randomly sampling a single
  % parameter value from theta0, and then computing the actual gradient
  % computed by fun, then estimating the gradient using the numerical gradient
  % checking algorithm, resulting g_est.
  delta=1e-3;
  sum_error=0;

  fprintf(' Iter       i             err');
  fprintf('           g_est               g               f\n')

  for i=1:num_checks
    T = theta0;
    j = randsample(numel(T),1);
    T0=T; T0(j) = T0(j)-delta;
    T1=T; T1(j) = T1(j)+delta;

    % varargin cell array serves the same purpose as *args in Python
    % constructs, where an arbitrary list of trailing arguments can be
    % unpacked in the function body.
    [f,g] = fun(T, varargin{:});
    f0 = fun(T0, varargin{:});
    f1 = fun(T1, varargin{:});

    g_est = (f1-f0) / (2*delta);
    error = abs(g(j) - g_est);

    fprintf('% 5d  % 6d % 15g % 15f % 15f % 15f\n', ...
            i,j,error,g(j),g_est,f);

    sum_error = sum_error + error;
  end

  average_error=sum_error/num_checks;
