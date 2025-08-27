#include <bits/stdc++.h>
using i64 = long long;

i64 mul(i64 a, i64 b, i64 m) {
    return static_cast<__int128>(a) * b % m;
}
i64 power(i64 a, i64 b, i64 p) {
    i64 res = 1 % p;
    for (; b; b >>= 1, a = mul(a, a, p))
        if (b & 1)
            res = mul(res, a, p);
    return res;
}

// Miller-Rabin
bool isprime(i64 n) {
    if (n < 2)
        return false;
    static constexpr int A[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    // A[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022LL};
    int s = __builtin_ctzll(n - 1);
    i64 d = (n - 1) >> s;
    for (auto a : A) {
        if (a == n)
            return true;
        i64 x = power(a, d, n);
        if (x == 1 || x == n - 1)
            continue;
        bool ok = false;
        for (int i = 0; i < s - 1; ++i) {
            x = mul(x, x, n);
            if (x == n - 1) {
                ok = true;
                break;
            }
        }
        if (!ok)
            return false;
    }
    return true;
}