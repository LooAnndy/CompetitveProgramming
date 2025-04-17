#include <bits/stdc++.h>
template <class Info>
struct SGT {
#define l(u) (u << 1)
#define r(u) (u << 1 | 1)
    int n;
    std::vector<Info> info;
    SGT() {}
    SGT(int _n) {
        std::vector<Info> _init(_n + 1, Info());
        init(_init);
    }
    template <class T>
    void init(const std::vector<T>& _init) {
        // build [1,n]
        n = _init.size() - 1;
        info.assign(n << 2, Info());
        auto build = [&](auto self, int l, int r, int u) {
            if (l == r) {
                info[u] = _init[l];
                return;
            }
            int m = (l + r) >> 1;
            self(self, l, m, l(u));
            self(self, m + 1, r, r(u));
            pushup(u);
        };
        build(build, 1, n, 1);
    }
    void pushup(int u) {
        info[u] = info[l(u)] + info[r(u)];
    }
    template <typename T>
    void update(int pos, int u, int l, int r, const T add) {
        if (l == pos && r == pos) {
            info[u] = info[u] + add;
            return;
        }
        int m = (l + r) >> 1;
        if (pos <= m) {
            update<T>(pos, l(u), l, m, add);
        }
        if (pos > m) {
            update<T>(pos, r(u), m + 1, r, add);
        }
        pushup(u);
    }
    Info query(int ql, int qr, int u, int l, int r) {
        if (qr < l || ql > r) {
            return Info();
        }
        if (ql <= l && r <= qr) {
            return info[u];
        }
        int m = (l + r) >> 1;
        return query(ql, qr, l(u), l, m) + query(ql, qr, r(u), m + 1, r);
    }
    int lower_search(int l, int r, int u, int find) {
        if (l == r) {
            return l;
        }
        int m = (l + r) >> 1;
        if (info[l(u)].max < find) {
            return lower_search(m + 1, r, r(u), find);
        }
        return lower_search(l, m, l(u), find);
    }
    int upper_search(int l, int r, int u, int find) {
        if (l == r) {
            return l;
        }
        int m = (l + r) >> 1;
        if (info[l(u)].max <= find) {
            return upper_search(m + 1, r, r(u), find);
        }
        return upper_search(l, m, l(u), find);
    }
#undef l
#undef r
};

struct Info {
    int max, min;
    Info(): max(std::numeric_limits<int>::min() / 2), min(std::numeric_limits<int>::max() / 2) {}
    Info(int val): max(val), min(val) {}
    Info(int _max, int _min): max(_max), min(_min) {}
    Info operator+(int x) {
        return Info(max + x, min + x);
    }
    Info operator+(const Info& rhs) {
        auto cmax = std::max(rhs.max, max);
        auto cmin = std::min(rhs.min, min);
        return Info(cmax, cmin);
    }
};