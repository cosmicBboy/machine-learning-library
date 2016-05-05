% plotting data!

t = [0: 0.01: 0.98]
y1 = sin(2 * pi * 4 * t);
y2 = cos(2 * pi * 4 * t);

plot(t, y1);
hold on; % plot multiple traces in same figure
plot(t, y2, 'r');
xlabel('time')
ylabel('value')
legend('sin', 'cos') % add legend
title('my plot')

% save plot to file
% print dpng 'plots/myplot.png'
close % remove figure

figure(1); plot(t, y1)
figure(2); plot(t, y2)
close

% subplots
subplot(1,2,1); % divides plot to 1x2 grid, access first element
plot(t, y1);
subplot(1,2,2); % access second element
plot(t, y2);
axis([0.5 1 -1 1]) % set xrange and y range
clf; % clears figure
close;

% heatmaps
A = magic(5)
imagesc(A) % plot as 5x5 grid of colors
imagesc(magic(15)), colorbar, colormap gray;
% close;

% you can comma chain commands
a = 1, b = 2, c = 3;
