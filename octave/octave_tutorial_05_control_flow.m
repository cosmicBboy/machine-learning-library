% for loops
v = zeros(10, 1)
for i = 1:10,
    v(i) = 2^i;
end;
v

% while loop
i = 1;
while i <= 5,
    v(i) = 100;
    i = i+1;
end;
v

% using a break statement
i = 1;
while true,
    v(i) = 999;
    i = i + 1;
    if i == 6,
        break;
    end;
end;
v

v(1) = 2
if v(1) == 1,
    disp('The value is one');
elseif v(1) == 2,
    disp('The value is two');
else,
    disp('The value is not one or two.');
end;

% square this number using square_this_number.m script
square_this_number(5)
[a, b] = square_and_cube_this_number(5)

X = [1 1; 1 2; 1 3] % training example features
y = [1; 2; 3] % training example labels
theta = [0; 1]; % set thetas
j = cost_function_j(X, y, theta) % compute cost

theta = [0; 0]; % set new thetas
j = cost_function_j(X, y, theta) % compute cost
