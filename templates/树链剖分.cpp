#include <bits/stdc++.h>
struct HLD {
    int n;
    std::vector<int> par, dep, top, son, siz;
    std::vector<std::vector<int>> G;

    HLD() {}
    HLD(int _n) {
        init(_n);
    }
    void init(int _n) {
        this->n = _n;
        par.resize(n + 1);
        dep.resize(n + 1);
        top.resize(n + 1);
        son.resize(n + 1);
        siz.resize(n + 1);
        G.resize(n + 1);
    }

    void addEdge(int u, int v) {
        G[u].push_back(v);
        G[v].push_back(u);
    }

    void work(int root = 1) {
        dfs1(root, root);
        dfs2(root, root);
    }

    void dfs1(int u, int fa) {
        siz[u] = 1;
        for (auto v : G[u]) {
            if (v == fa) continue;
            par[v] = u;
            dep[v] = dep[u] + 1;
            dfs1(v, u);
            siz[u] += siz[v];
            if (siz[v] > siz[son[u]]) {
                son[u] = v;
            }
        }
    }

    void dfs2(int u, int topf) {
        top[u] = topf;
        if (son[u]) {
            dfs2(son[u], topf);
        }
        for (auto v : G[u]) {
            if (son[u] == v || par[u] == v) continue;
            dfs2(v, v);
        }
    }

    int lca(int x, int y) {
        while (top[x] != top[y]) {
            if (dep[top[x]] > dep[top[y]]) {
                x = par[top[x]];
            } else {
                y = par[top[y]];
            }
        }
        return dep[x] < dep[y] ? x : y;
    }

    int dist(int u, int v) {
        return dep[u] + dep[v] - 2 * dep[lca(u, v)];
    }
};