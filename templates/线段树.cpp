#include <bits/stdc++.h>
#define lson u << 1
#define rson u << 1 | 1

using namespace std;
using ll = long long;

const int N = 1e6 + 10;
const ll INF = 1e18;
int n, q;
ll a[N];

struct Node {
    int l, r;
    ll covertag = -INF, addtag = 0;
    ll maxn = -INF;
    // 表示不需要下放懒标记
    bool push = 0;
} tr[N << 2];

void pushup(int u) {
    tr[u].maxn = max(tr[lson].maxn, tr[rson].maxn);
}

void pushdown(int l, int r, int u) {
    if (tr[u].covertag != -INF) {
        tr[lson].covertag = tr[u].covertag;
        tr[lson].maxn = tr[u].covertag;
        tr[rson].covertag = tr[u].covertag;
        tr[rson].maxn = tr[u].covertag;

        tr[lson].addtag = tr[rson].addtag = 0;
        tr[lson].push = tr[rson].push = 1;
        tr[u].covertag = -INF;
    }
    if (tr[u].addtag) {
        tr[lson].addtag += tr[u].addtag;
        tr[lson].maxn += tr[u].addtag;
        tr[rson].addtag += tr[u].addtag;
        tr[rson].maxn += tr[u].addtag;

        tr[lson].push = tr[rson].push = 1;
        tr[u].addtag = 0;
    }
    tr[u].push = 0;
}

void build(int l, int r, int u) {
    tr[u].l = l;
    tr[u].r = r;
    if (l == r) {
        tr[u].maxn = a[l];
        return;
    }
    int m = (l + r) >> 1;
    build(l, m, lson);
    build(m + 1, r, rson);
    pushup(u);
}

ll query(int l, int r, int u) {
    if (l <= tr[u].l && r >= tr[u].r) {
        return tr[u].maxn;
    }

    if (tr[u].push) {
        pushdown(l, r, u);
    }

    int m = (tr[u].l + tr[u].r) >> 1;
    ll ans = -INF;
    if (l <= m) {
        ans = max(ans, query(l, r, lson));
    }
    if (r > m) {
        ans = max(ans, query(l, r, rson));
    }
    return ans;
}

void add(int l, int r, ll c, int u) {
    if (l <= tr[u].l && r >= tr[u].r) {
        tr[u].maxn += c;
        tr[u].addtag += c;
        tr[u].push = 1;
        return;
    }
    if (tr[u].push) {
        pushdown(l, r, u);
    }
    int m = (tr[u].l + tr[u].r) >> 1;
    if (l <= m) {
        add(l, r, c, lson);
    }
    if (r > m) {
        add(l, r, c, rson);
    }
    pushup(u);
}

void cover(int l, int r, ll c, int u) {
    if (l <= tr[u].l && r >= tr[u].r) {
        tr[u].maxn = c;
        tr[u].addtag = 0;
        tr[u].covertag = c;
        tr[u].push = 1;
        return;
    }
    if (tr[u].push) {
        pushdown(l, r, u);
    }
    int m = (tr[u].l + tr[u].r) >> 1;
    if (l <= m) {
        cover(l, r, c, lson);
    }
    if (r > m) {
        cover(l, r, c, rson);
    }
    pushup(u);
}

void debug() {
    for (int i = 1; i <= n << 1; i++) {
        cout << "l:" << tr[i].l << " r:" << tr[i].r << " maxn:" << tr[i].maxn << " covertag:" << tr[i].covertag << " addtag:" << tr[i].addtag << " push:" << tr[i].push << endl;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin >> n >> q;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
    }
    build(1, n, 1);

    while (q--) {
        int op, l, r;
        ll c;
        cin >> op >> l >> r;
        if (op == 1) {
            cin >> c;
            cover(l, r, c, 1);
        } else if (op == 2) {
            cin >> c;
            add(l, r, c, 1);
        } else {
            cout << query(l, r, 1) << '\n';
        }
        // debug();
    }
    return 0;
}