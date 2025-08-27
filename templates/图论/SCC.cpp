#include <bits/stdc++.h>
struct SCC {
    int n;                             // 节点数
    std::vector<std::vector<int>> adj; // 图的邻接表
    std::vector<int> stk;              // 栈，用于 Tarjan 算法
    std::vector<int> dfn, low, bel;    // dfn: 访问顺序, low: 最小可回溯 dfn, bel: 节点所属 SCC 编号
    int cur, cnt;                      // cur: 当前时间戳, cnt: SCC 编号计数

    SCC() {}
    SCC(int n) {
        init(n);
    }

    void init(int n) {
        this->n = n;
        adj.assign(n, {});
        dfn.assign(n, -1);
        low.resize(n);
        bel.assign(n, -1);
        stk.clear();
        cur = cnt = 0;
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
    }

    void dfs(int x) {
        dfn[x] = low[x] = cur++;
        stk.push_back(x);

        for (auto y : adj[x]) {
            if (dfn[y] == -1) {
                dfs(y);
                low[x] = std::min(low[x], low[y]);
            } else if (bel[y] == -1) {
                low[x] = std::min(low[x], dfn[y]);
            }
        }

        if (dfn[x] == low[x]) {
            int y;
            do {
                y = stk.back();
                bel[y] = cnt;
                stk.pop_back();
            } while (y != x);
            cnt++;
        }
    }

    const std::vector<int>& work() {
        for (int i = 0; i < n; i++) {
            if (dfn[i] == -1) {
                dfs(i);
            }
        }
        return bel;
    }
};