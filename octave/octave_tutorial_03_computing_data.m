A = [1 2; 3 4; 5 6]
B = [11 12; 13 14; 15 16]
C = [1 1; 2 2]

% matrix multiplication
A * C

% element-wise operations
A .* B % multiplication
A .^ 2 % square

v = [1; 2; 3]
1 ./ v % reciprocal
exp(v) % base-e exponentiation
abs(v) % absolutve value
-v % negative values
v + ones(length(v), 1) % increment vector by one
v + 1 % another way to increment vector by one

% transpose
A'
(A')'

% max values
a = [1 15 2 0.5]
[val, ind] = max(a)
max(A) % column-wise maximum
a < 3 % element-wise conditionals, returns boolean vector
find(a < 3) % which of the elements in boolean vector are true?

% the magic function
A = magic(3) % "magic squares", all of rows and columns sum up to same value
[r, c] = find(A >= 7)

% some vector operations
sum(a) % add all elements of vector
prod(a) % product of all elements
floor(a) % round down to nearest integer
ceil(a) % round up to nearest integer
max(rand(3), rand(3)) % element wise maximum of two 3x3 random matrices
max(A, [], 1) % take the column-wise maximum of a matrix
max(A, [], 2) % take the row-wise maximum of a matrix
max(A(:)) % turn A into a vector, find maximum values

% check that magic matrix is what it claims to be
A = magic(9)
sum(A, 1) % column-wise sum of matrix
sum(A, 2) % row-wise sum of matrix
A .* eye(9) % element-wise multiply A with identity matrix
sum((A .* eye(9))(:)) % sum up the diagonal

% flip a matrix up-down
flipud(eye(9))

A = magic(3)
tmp = pinv(A) % pseudo-inverse (aka inverse)
round(A * tmp) % check that A * pinv(A) == A
