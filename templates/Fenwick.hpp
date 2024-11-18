#include <bits/stdc++.h>

template <typename T>
struct Fenwick {
    int n;
    std::vector<T> tr;

    Fenwick() {};
    Fenwick(int _n = -1) {
        n = _n + 1;
        tr.resize(n + 1);
    };

    T presum(int x) {
        T res {};
        for (int i = x; i >= 1; i -= i & (-i)) {
            res = res + tr[i];
        }
        return res;
    }

    // sum of (x , y]
    T rangesum(int l, int r) {
        if (l > r)
            std::swap(l, r);
        return presum(r) - presum(l);
    }

    void add(int x, T c) {
        for (int i = x; i <= n; i += i & (-i)) {
            tr[i] = tr[i] + c;
        }
    }
};