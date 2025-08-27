#include <bits/stdc++.h>

struct EDcc {
    int n, m, idx, stamp;
    std::vector<int> e, ne, h;
    std::vector<int> dfn, low, stk;
    std::vector<char> bridge; // 判断边是不是桥
    std::vector<std::vector<int>> eDc;

    EDcc() {}
    EDcc(int n, int m) {
        init(n, m);
    }

    void init(int n, int m) {
        this->n = n;
        this->m = m;
        idx = -1;
        stamp = -1;
        h.assign(n, -1);
        dfn.assign(n, -1);
        low.assign(n, -1);

        ne.assign(m * 2, -1);
        e.assign(m * 2, -1);
        bridge.assign(m * 2, -1);

        stk.clear();
        eDc.clear();
    }

    void addEdge(int u, int v) {
        e[++idx] = v;
        ne[idx] = h[u];
        h[u] = ++idx;
    }

    void tarjan(int u, int lst) {
        dfn[u] = low[u] = ++stamp;
        stk.push_back(u);

        for (int i = h[u]; i != -1; i = ne[i]) {
            int v = e[i];
            if ((i ^ 1) == lst) continue;
            if (dfn[v] == -1) {
                tarjan(v, i);
                low[u] = std::min(low[u], low[v]);
                if (low[v] > dfn[u]) {
                    bridge[i] = bridge[i ^ 1] = 1;
                }
            } else {
                low[u] = std::min(low[u], dfn[v]);
            }
        }
        if (dfn[u] == low[u]) {
            std::vector<int> vc;
            int tmp;
            do {
                tmp = stk.back();
                stk.pop_back();
                vc.push_back(tmp);
            } while (tmp != u);
            eDc.push_back(vc);
        }
    }

    void work() {
        for (int i = 0; i < n; i++) {
            if (dfn[i] == -1) {
                tarjan(i, -1);
            }
        }
    }
};