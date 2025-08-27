#include <bits/stdc++.h>
using u64 = unsigned long long;
struct LinearBasis {
    static constexpr int K = 60;
    std::array<u64, K> b;

    LinearBasis() {
        b.fill(0);
    }

    bool insert(u64 x) {
        for (int i = K - 1; i >= 0; i--) {
            if (!(x >> i & 1)) continue;
            if (!b[i]) {
                b[i] = x;
                return true;
            }
            x ^= b[i];
        }
        return false;
    }

    u64 ask(u64 x = 0) {
        for (int i = K - 1; i >= 0; i--) {
            if ((x ^ b[i]) > x) {
                x ^= b[i];
            }
        }
        return x;
    }
};