k = 1 : 100;
eps_k = zeros(5, 100);
eps_k(1, :) = 1 ./ k;
eps_k(2, :) = 100 ./ (100 + k);
eps_k(3, :) = (1 + log(k)) ./ k;
eps_k(4, :) = (1 + 5 * log(k)) ./ k;
eps_k(5, :) = 200 ./ (200 + k);
figure
title('Comparison of epsilon')
hold on
for i = 1 : size(eps_k, 1)
    plot(k, eps_k(i, :), 'linewidth', 2)
end
legend('1 / k', '100 / (100 + k)', '(1 + log(k)) / k', '(1 + 5 * log(k)) / k', '200 / (200 + k)')
hold off