## 数学
### exgcd
```c++
int exgcd(int a, int b, int& x, int& y) {
    if (!b) {
        x = 1;
        y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
```
### 线性筛
```c++
int n, primes[N], cnt = 0;
bool st[N];

void euler(int n) {
    for (int i = 2; i < n; i++) {
        if (!st[i]) primes[cnt++] = i;
        for (int j = 0; primes[j] <= n / i; j++) {
            st[primes[j] * i] = 1;
            if (i % primes[j] == 0) break;
        }
    }
    st[0] = st[1] = 1;
}
```
### 线性基(异或线性基)
1.普通消元(贪心)
```c++
ull p[64];
void insert(ull x) {
    for (int i = 63; ~i; --i) {
        if (!(x >> i & 1)) continue;
        if (!p[i]) {
            p[i] = x;
            break;
        }
        x ^= p[i];
    }
}
```
2.高斯消元
```c++
int row = 1;
for (int col = 63; col >= 0 && row <= n; col--) {
    for (int i = row; i <= n; i++) {
        if (a[i] >> col & 1) {
            std::swap(a[row], a[i]);
            break;
        }
    }
    if (!(a[row] >> col & 1)) continue;
    for (int i = 1; i <= n; i++) {
        if (i == row) continue;
        if (a[i] >> col & 1) {
            a[i] ^= a[row];
        }
    }
    row++;
}
--row;
```
#### 查询异或最值
贪心，从最高（最低位）开始异或，每一轮都和ans取个max(min)
#### 查询k小值
高斯消元后能保证每一个位数上的1都是独特的，所以说查询第k个元素的时候直接
对k进行二进制分解对于每一位上的1，去找线性基上唯一的1对应的数异或即可。
注意对0的特判（没有0应该查询x-1）（线性基相比于原来的大小变小就是可以有0）
## 图论
### 树
1.重心
性质：1.以树的重心为根时，所有子树的大小都不超过整棵树大小的一半，这个是充要条件（重心至多两个，并且相邻）</br>
2.树中所有点到某个点的距离和中，到重心的距离和是最小的；如果有两个重心，那么到它们的距离和一样。
2.树链剖分
```c++
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
```
### 强连通分量
```c++
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
```
### 点双

### 边双

### 网络流
最大流
```c++
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
```
最小费用最大流，SPFA控制费用
```c++
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
```

## 字符串
先介绍一下$\pi$函数
$$\pi[i] = \max_ {k = 0 \dots i} \{k : s[0 \dots k-1] = s[i-(k-1) \dots i] \}$$
其中特殊地，$\pi[0]=0$
```c++
std::vector<int> pre_pi(std::string s) {
  int n = (int)s.length();
  vector<int> pi(n);
  for (int i = 1; i < n; i++) {
    int j = pi[i - 1];
    while (j > 0 && s[i] != s[j]) {
         j = pi[j - 1];
    }
    if (s[i] == s[j]) j++;
    pi[i] = j;
  }
  return pi;
}
```
### kmp
不妨把两个字符串连接在一起，利用$\pi$函数即可解决
```c++
//return positions that successfully matched
vector<int> find_occurrences(string text, string pattern) {
  string cur = pattern + '#' + text;
  int sz1 = text.size(), sz2 = pattern.size();
  vector<int> v;
  vector<int> lps = prefix_function(cur);
  for (int i = sz2 + 1; i <= sz1 + sz2; i++) {
    if (lps[i] == sz2) v.push_back(i - 2 * sz2);
  }
  return v;
}
```
### Z函数（扩展kmp）
$z[i]$匹配的是s和s[i:]的最长公共前缀
```c++
vector<int> z_function(string s) {
  int n = (int)s.length();
  vector<int> z(n);
  for (int i = 1, l = 0, r = 0; i < n; ++i) {
    if (i <= r && z[i - l] < r - i + 1) {
      z[i] = z[i - l];
    } else {
      z[i] = max(0, r - i + 1);
      while (i + z[i] < n && s[z[i]] == s[i + z[i]]) ++z[i];
    }
    if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
  }
  return z;
}
```
### AC自动机
现在还不会捏

### Manacher
函数会把原先字符串两个数之间（包括头尾）插入'#'，d[i]表示以位置i为中心的最长回文字符串有多长
```c++
std::vector<int> Manacher(std::string& t) {
    int n = t.size();
    string s;
    s.resize(n * 2 + 1);
    for (int i = 0; i < n; i++) {
        s[i << 1] = '#';
        s[i << 1 | 1] = t[i];
    }
    s[2 * n] = '#';
    n = 2 * n + 1;
    std::vector<int> d(n);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 1 : std::min(d[l + r - i], r - i + 1);
        while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) {
            k++;
        }
        d[i] = --k;
        if (i + k > r) {
            l = i - k;
            r = i + k;
        }
    }
    return d;
}
```
## 数据结构
### 树状数组
```c++
template <typename T>
struct Fenwick {
    int n;
    std::vector<T> tr;

    Fenwick() {};
    Fenwick(int _n) {
        this->n = _n;
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
```
### 线段树
```c++
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
```
### 懒线段树
```c++
template <class Info, class Tag>
struct LSGT {
#define l(u) (u << 1)
#define r(u) (u << 1 | 1)
    int n;
    std::vector<Info> info;
    std::vector<Tag> tag;
    LSGT() {}
    LSGT(int _n) {
        std::vector<Info> _init(_n + 1, Info());
        init(_init);
    }
    template <class T>
    void init(const std::vector<T>& _init) {
        // build [1,n]
        n = _init.size() - 1;
        info.assign(n << 2, Info());
        tag.assign(n << 2, Tag());
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
    void pushdown(int u) {
        tag[l(u)].apply(tag[u]), info[l(u)].apply(tag[u]);
        tag[r(u)].apply(tag[u]), info[r(u)].apply(tag[u]);
        tag[u] = Tag();
    }
    template <typename T>
    void update(int ql, int qr, int u, int l, int r, const T add) {
        if (ql <= l && r <= qr) {
            info[u] = info[u] + add;
            tag[u] = tag[u] + add;
            return;
        }
        int m = (l + r) >> 1;
        pushdown(u);
        if (ql <= m) {
            update<T>(ql, qr, l(u), l, m, add);
        }
        if (qr > m) {
            update<T>(ql, qr, r(u), m + 1, r, add);
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
        pushdown(u);
        return query(ql, qr, l(u), l, m) + query(ql, qr, r(u), m + 1, r);
    }
    int lower_search(int l, int r, int u, int find) {
        if (l == r) {
            return l;
        }
        int m = (l + r) >> 1;
        pushdown(u);
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
        pushdown(u);
        if (info[l(u)].max <= find) {
            return upper_search(m + 1, r, r(u), find);
        }
        return upper_search(l, m, l(u), find);
    }
#undef l
#undef r
};

struct Tag {
    int x;
    Tag() {}
    Tag(int _x): x(_x) {};
    Tag operator+(int add) {
        return Tag(x + add);
    }
    void apply(const Tag& p) {
        x += p.x;
    }
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
    void apply(Tag& t) {
        max += t.x, min += t.x;
    }
};
```
### 并查集
v1 维护大小的并查集
```c++
struct DSU {
    std::vector<int> f, siz;

    DSU() {}
    DSU(int n) {
        init(n);
    }

    void init(int n) {
        f.resize(n);
        std::iota(f.begin() + 1, f.end(), 1);
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
```
带权并查集（维护到根节点位置）
```c++
struct WeightDSU {
    /*
    siz维护子树的大小
    dist根到当前点的距离
    */
    std::vector<int> f, dist, dep;

    WeightDSU() {}
    WeightDSU(int n) {
        init(n);
    }

    void init(int n) {
        f.resize(n);
        std::iota(f.begin() + 1, f.end(), 1);
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
        dep[rootx] = std::max(dep[rootx], dep[y] + dist[x] + 1);
        dist[y] = dist[x] + 1;
        f[y] = rootx;
        return true;
    }
};
```
## 计算几何

## 杂项

### 点分治

### 

### 莫队
就是分块离线
洛谷P1494直接
```c++
struct Query {
    int l, r, id;
};

void solve() {
    int n, m;
    std::cin >> n >> m;
    std::vector<int> col(n + 1), belong(n + 1), cnt(n + 1);
    std::vector<Query> q(m + 1);
    int block = std::max(10, (int) sqrt(n));
    for (int i = 1; i <= n; i++) {
        std::cin >> col[i];
        belong[i] = i / block;
    }
    for (int i = 1; i <= m; i++) {
        std::cin >> q[i].l >> q[i].r;
        q[i].id = i;
    }
    LL res = 0;
    auto add = [&](int c) -> void {
        res += 1LL * cnt[c];
        cnt[c]++;
    };
    auto sub = [&](int c) -> void {
        cnt[c]--;
        res -= 1LL * cnt[c];
    };

    std::sort(q.begin() + 1, q.end(), [&](const Query& a, const Query& b) {
        if (belong[a.l] == belong[b.l]) {
            if (belong[a.l] & 1) return a.r > b.r;
            return a.r < b.r;
        }
        return belong[a.l] < belong[b.l];
    });
    std::vector<std::pair<LL, LL>> ans(m + 1);
    for (int l = 1, r = 0, i = 1; i <= m; i++) {
        if (q[i].l == q[i].r) {
            ans[q[i].id] = { 0, 1 };
            continue;
        }
        while (l > q[i].l) {
            add(col[--l]);
        }
        while (r < q[i].r) {
            add(col[++r]);
        }
        while (l < q[i].l) {
            sub(col[l++]);
        }
        while (r > q[i].r) {
            sub(col[r--]);
        }
        ans[q[i].id] = { res, 1LL * (r - l) * (r - l + 1) / 2 };
    }

    for (int i = 1; i <= m; i++) {
        auto [num, deno] = ans[i];
        if (num == 0) {
            std::cout << 0 << "/" << 1 << '\n';
        } else {
            auto g = std::__gcd(num, deno);
            std::cout << num / g << '/' << deno / g << '\n';
        }
    }
}
```