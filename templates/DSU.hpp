#include <bits/stdc++.h>

struct DSU {
    std::vector<int> f, siz;

    DSU() {}
    DSU(int n) {
        init(n);
    }

    void init(int n) {
        f.resize(n);
        std::iota(f.begin(), f.end(), 0);
        siz.assign(n, 1);
    }

    int find(int x) {
        while (x != f[x]) {
            x = f[x] = f[f[x]];
        }
        return x;
    }

    bool same(int x, int y) {
        return find(x) == find(y);
    }

    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) {
            return false;
        }
        siz[x] += siz[y];
        f[y] = x;
        return true;
    }

    int size(int x) {
        return siz[find(x)];
    }
};

/*
//维护树根合并和距离维护
struct DSU {
    //siz维护子树的大小
    //dist根到当前点的距离
    std::vector<int> f, dist, dep;

    DSU() {}
    DSU(int n) {
        init(n);
    }

    void init(int n) {
        f.resize(n);
        std::iota(f.begin(), f.end(), 0);
        dist.assign(n, 0);
        dep.assign(n, 0);
    }

    int find(int x) {
        if (f[x] != x) {
            int t = f[x];
            f[x] = find(f[x]);
            dist[x] += dist[t];
        }
        return f[x];
    }

    bool same(int x, int y) {
        return find(x) == find(y);
    }

    bool merge(int x, int y) {
        int rootx = find(x);
        y = find(y);
        if (rootx == y) {
            return false;
        }
        dep[rootx] = max(dep[rootx], dep[y] + dist[x] + 1);
        dist[y] = dist[x] + 1;
        f[y] = rootx;
        return true;
    }
};
*/