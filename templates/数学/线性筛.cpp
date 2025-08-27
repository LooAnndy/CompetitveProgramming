#include <bits/stdc++.h>
struct Prime {
    std::vector<int> primes, minp, mu, phi;

    Prime(int n) {
        sieve(n);
    }

    void sieve(int n) {
        minp.assign(n + 1, 0);
        mu.assign(n + 1, 0);
        phi.assign(n + 1, 0);
        primes.clear();

        mu[1] = phi[1] = 1;
        for (int i = 2; i <= n; i++) {
            if (minp[i] == 0) {
                primes.push_back(i);
                minp[i] = i;
                mu[i] = -1;
                phi[i] = i - 1;
            }

            for (auto p : primes) {
                if (i * p > n) {
                    break;
                }
                minp[i * p] = p;
                if (i % p == 0) {
                    mu[i] = 0;
                    phi[i * p] = phi[i] * p;
                    break;
                }
                mu[i * p] = -mu[i];
                phi[i * p] = phi[i] * (p - 1);
            }
        }
    }
};