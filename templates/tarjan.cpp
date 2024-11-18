#include <bits/stdc++.h>
const int N = 10086;
int scc = 0;
int dfn[N] = { 0 }, low[N] = { 0 }, stamp = 0;
int id[N];
std::vector<int> G[N];
void tarjan(int u) {
    static int stk[N], top = -1;
    static bool in_stk[N];

    dfn[u] = low[u] = ++stamp;
    stk[++top] = u, in_stk[u] = 1;
    for (auto v : G[u]) {
        if (!dfn[v]) {
            tarjan(v);
            low[u] = std::min(low[u], low[v]);
        } else if (in_stk[v]) {
            low[u] = std::min(low[u], dfn[v]);
        }
    }

    if (dfn[u] == low[u]) {
        ++scc;
        int y;
        do {
            y = stk[top--];
            in_stk[y] = 0;
            id[y] = scc;
        } while (y != u);
    }
}