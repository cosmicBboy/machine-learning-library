% generate matrix
A = [1 2; 3 4; 5 6]

% get size of matrix
size(A)
sz = size(A) % returns a matrix
sz2 = size(sz) % also returns a matrix

size(A, 1) % size of first dimension
size(A, 2) % size of second dimension

% length returns size of longest dimension
v = [1 2 3 4]
length(v)
length(A)

pwd % print working directory
% `cd '/Users/Niels/'` % change directory
% `ls` to list files in current working directory

%% READ IN DATA
load 'data/featuresX.dat'
load 'data/priceY.dat'

disp('X features')
size(featuresX)
disp(featuresX)
disp('price data')
size(priceY)
disp(priceY)

% list current variable environment
whos

% delete variable
clear featuresX
whos

% SUBSETTING A VECTOR
v = priceY(1:4) % select first 10 elements
whos

% SAVING DATA TO .mat FILE
save 'data/hello.mat' v; % saves data in binary format

% RELOADING VARIABLES
clear
load 'data/hello.mat'
whos

% SAVE DATA AS TEXT FILES
save 'data/hello.txt' v -ascii;

% SUBSETTING A MATRIX
A = [1 2; 3 4; 5 6]
A(3,2) % get 3rd row, 2nd column
A(2,:) % get 2nd row, all columns
A(:,2) % get all rows, 2nd column
A([1 3], 2) % get 1st and 3rd row, 2nd column

% MATRIX VALUE ASSIGNMENT

A(:, 2) = [10; 11; 12] % replace 2nd column with new values
A = [ A, [100; 101; 102]] % column concat
size(A)

% put all elements of A into a single vector
A(:)

% column concat matrices
A = [1 2; 3 4; 5 6]
B = [11 12; 13 14; 15 16]
C = [A B]

% row concatenate matrices
D = [A; B]
