#include <bits/stdc++.h>
struct MCF {
    int n, m, idx;
    std::vector<int> h, ne, e;
    std::vector<int> pre, flow;
    std::vector<int> dis, cap, cost;

    MCF(int _n, int _m): idx(-1) {
        init(_n, _m);
    }

    void init(int _n, int _m) {
        this->n = _n, this->m = _m;
        h.resize(n, -1);
        ne.resize(m << 1);
        e.resize(m << 1);

        pre.resize(m << 1);
        flow.resize(n);

        dis.resize(n);
        cap.resize(m << 1);
        cost.resize(m << 1);
    }

    void addEdge(int u, int v, int vol, int w) {
        add(u, v, vol, w);
        add(v, u, 0, -w);
    }

    bool spfa(int st, int ed) {
        // shortest path
        dis.assign(n, 0x3f3f3f3f);
        flow.assign(n, 0x3f3f3f3f);
        std::vector<char> vis(n);
        std::queue<int> q;
        q.push(st);
        vis[st] = 1, dis[st] = 0;
        while (!q.empty()) {
            auto u = q.front();
            q.pop();
            vis[u] = 0;
            for (int i = h[u]; ~i; i = ne[i]) {
                int v = e[i];
                if (!cap[i]) continue;
                if (dis[v] > dis[u] + cost[i]) {
                    dis[v] = dis[u] + cost[i];
                    flow[v] = std::min(flow[u], cap[i]);
                    pre[v] = i;
                    if (!vis[v]) {
                        q.push(v);
                        vis[v] = 1;
                    }
                }
            }
        }
        return dis[ed] != 0x3f3f3f3f;
    }

    void update(int st, int ed) {
        int u = ed;
        while (u != st) {
            int i = pre[u];
            cap[i] -= flow[ed];
            cap[i ^ 1] += flow[ed];
            u = e[i ^ 1];
        }
    }

    std::pair<int, int> work(int st, int ed) {
        int maxflow = 0, res_cost = 0;
        while (spfa(st, ed)) {
            update(st, ed);
            maxflow += flow[ed];
            res_cost += flow[ed] * dis[ed];
        }
        return { maxflow, res_cost };
    }

private:
    void add(int a, int b, int vol, int w) {
        e[++idx] = b;
        cap[idx] = vol;
        cost[idx] = w;
        ne[idx] = h[a];
        h[a] = idx;
    }
};