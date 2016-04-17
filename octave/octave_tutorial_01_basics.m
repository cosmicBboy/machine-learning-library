% BASIC COMMANDS

% add
5 + 6

% subtract
3 - 2

% multiply
5 * 10

% divide
1 / 2

% exponents
2 ^ 6

% conditionals
1 == 2 % 1 equals 2
1 ~= 2 % 1 not equals 2
1 && 0 % AND operator
1 || 0 % OR operator
xor(1, 0) % XOR operator

% if you want to format the prompt string
% PS1('>> ')

% colons suppress output in a REPL session
a = 3;

% pi...
a = pi

% more complex printing
disp(a)
disp('')
disp('pi to two decimal places')
disp(sprintf('2 decimals: %0.2f', a))
disp('')
disp('pi to 6 decimal places')
disp(sprintf('6 decimals: %0.6f', a))

% matrices
A = [1 2;
     3 4;
     5 6]

% row vector
v = [1 2 3]

% column vector
v = [1; 2; 3;]

% vector number ranges
% start_number : increment_by : end_number
v = 1:0.1:2
v = 1:6

% matrix of ones
ones(2,3) % rows, columns
2 * ones(2,3)

% matrix of random numbers
rand(3,3) % rows, columns

% gaussian/normal random sample
w = randn(1,3)
w = -6 + sqrt(10) * randn(1,10000);

% plotting a histogram
f = figure()
hist(w, 50) % with 50 bins
[1 2 3]

% identity matrix
I = eye(4)
I = eye(6)
% `help eye`to get into on eye
