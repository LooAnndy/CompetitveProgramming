#include <bits/stdc++.h>
using i64 = long long;

// 对任意模数p的逆元递推
// std::vector<i64> inv(p);
// inv[1] = 1;
// for (i64 i = 2; i < p; i++) {
//     inv[i] = (p - (p / i) * 1LL * inv[p % i] % p) % p;
// }

constexpr i64 Mod = 998244353;

constexpr i64 qmi(i64 a, i64 b) {
    a %= Mod;
    i64 res = 1LL;
    for (; b; b >>= 1) {
        if (b & 1) res = res * a % Mod;
        a = a * a % Mod;
    }
    return res;
}

constexpr i64 inv(i64 a) {
    return qmi(a, Mod - 2);
}

struct Comb {
    int n;
    std::vector<i64> _fac;
    std::vector<i64> _invfac;
    std::vector<i64> _inv;

    Comb(): n { 0 }, _fac { 1 }, _invfac { 1 }, _inv { 0 } {}

    Comb(int n): Comb() {
        init(n);
    }

    void init(int m) {
        if (m <= n) return;
        _fac.resize(m + 1);
        _invfac.resize(m + 1);
        _inv.resize(m + 1);

        for (int i = n + 1; i <= m; i++) {
            _fac[i] = _fac[i - 1] * i % Mod;
        }
        _invfac[m] = qmi(_fac[m], Mod - 2);
        for (int i = m; i > n; i--) {
            _invfac[i - 1] = _invfac[i] * i % Mod;
            _inv[i] = _invfac[i] * _fac[i - 1] % Mod;
        }
        n = m;
    }

    i64 fac(int m) {
        if (m > n) init(2 * m);
        return _fac[m];
    }

    i64 invfac(int m) {
        if (m > n) init(2 * m);
        return _invfac[m];
    }

    i64 inv(int m) {
        if (m > n) init(2 * m);
        return _inv[m];
    }

    i64 binom(int n, int m) {
        if (n < m || m < 0) return 0;
        if (n > 1e7) {
            i64 ini { 1 };
            for (int i = n - m + 1; i <= n; i++) {
                ini = ini * 1LL * i % Mod;
            }
            return ini * invfac(m) % Mod;
        }
        return fac(n) * invfac(m) % Mod * invfac(n - m) % Mod;
    }
} comb(1e6 + 50);