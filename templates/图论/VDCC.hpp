#include <bits/stdc++.h>

struct VDcc {
    int n, stamp;
    std::vector<int> dfn, low, stk, cut;
    std::vector<std::vector<int>> adj, vDc;

    VDcc() {}
    VDcc(int n) {
        init(n);
    }

    void init(int n) {
        this->n = n;
        stamp = -1;
        dfn.assign(n, -1);
        low.assign(n, -1);
        cut.assign(n, 0);
        adj.assign(n, {});

        stk.clear();
        vDc.clear();
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
    }

    void tarjan(int u, int anc) {
        dfn[u] = low[u] = ++stamp;
        if (adj[u].size() == 0) {
            vDc.push_back({ u });
            return;
        }
        stk.push_back(u);
        int child = 0;
        for (auto v : adj[u]) {
            if (dfn[v] == -1) {
                child++;
                tarjan(v, anc);
                low[u] = std::min(low[u], low[v]);
                if (low[v] >= dfn[u]) {
                    if (u != anc) {
                        cut[u] = 1;
                    }
                    std::vector<int> vc;
                    int tmp;
                    do {
                        tmp = stk.back();
                        stk.pop_back();
                        vc.push_back(tmp);
                    } while (v != tmp);
                    vc.push_back(u);
                    vDc.push_back(vc);
                }
            } else {
                low[u] = std::min(low[u], dfn[v]);
            }
        }
        if (child >= 2 && u == anc) cut[u] = 1;
    }

    const std::vector<std::vector<int>>& work() {
        for (int i = 0; i < n; i++) {
            if (dfn[i] == -1) {
                tarjan(i, i);
            }
        }
        return vDc;
    }

    // 圆方树
    std::vector<std::vector<int>> build_bct() {
        std::vector<std::vector<int>> bct(n + vDc.size());
        int cnt = n;
        for (auto const& vc : vDc) {
            for (int u : vc) {
                bct[cnt].push_back(u);
                bct[u].push_back(cnt);
            }
            ++cnt;
        }
        return bct;
    }
};