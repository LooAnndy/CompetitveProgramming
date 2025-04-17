#include <bits/stdc++.h>
using LL = long long;
struct MaxFlow {
    int n, m, idx;
    std::vector<int> e, ne, h;
    std::vector<int> cur, dep;
    std::vector<LL> val;

    MaxFlow(int _n, int _m) {
        init(_n, _m);
    }

    void init(int _n, int _m) {
        this->n = _n;
        this->m = _m;
        idx = -1;
        e.resize(m << 1);
        ne.resize(m << 1);
        h.resize(n, -1);
        val.resize(m << 1);
        cur.resize(n);
        dep.resize(n);
    }

    void addEdge(int from, int to, LL w) {
        e[++idx] = to;
        val[idx] = w;
        ne[idx] = h[from];
        h[from] = idx;
    }

    bool bfs(int st, int ed) {
        dep.assign(n, -1);
        std::queue<int> q;
        q.push(st);
        dep[st] = 0;
        while (!q.empty()) {
            auto u = q.front();
            q.pop();
            cur[u] = h[u];
            for (int i = h[u]; ~i; i = ne[i]) {
                int v = e[i];
                if (dep[v] == -1 && val[i]) {
                    dep[v] = dep[u] + 1;
                    if (v == ed) return true;
                    q.push(v);
                }
            }
        }
        return false;
    }

    LL dfs(int ed, int u, LL limit) {
        if (u == ed) return limit;
        LL flow = 0;
        for (int i = cur[u]; ~i; i = ne[i]) {
            cur[u] = i;
            int v = e[i];
            if (dep[v] == dep[u] + 1 && val[i]) {
                LL find_flow = dfs(ed, v, std::min(limit - flow, val[i]));
                val[i] -= find_flow;
                val[i ^ 1] += find_flow;
                flow += find_flow;
                if (flow == limit)
                    return flow;
            }
        }
        return flow;
    }

    LL work(int s, int t) {
        LL maxflow = 0;
        while (bfs(s, t)) {
            maxflow += dfs(t, s, std::numeric_limits<LL>::max() / 2);
            maxflow = std::min(maxflow, std::numeric_limits<LL>::max() / 2);
        }
        return maxflow;
    }
};