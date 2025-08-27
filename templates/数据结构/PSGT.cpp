#include <bits/stdc++.h>
constexpr int N = 2E5 + 10;

int a[N], b[N], root[N];

namespace PSGT {
    struct Node {
        int ls, rs, cnt;
    };
    int idx = 0;
    Node tr[N << 5];

    void insert(int l, int r, int pre, int pos, int& u) {
        u = ++idx;
        tr[u] = tr[pre];
        tr[u].cnt++;
        if (l == r) {
            return;
        }
        int m = (l + r) >> 1;
        if (pos <= m) {
            insert(l, m, tr[pre].ls, pos, tr[u].ls);
        } else {
            insert(m + 1, r, tr[pre].rs, pos, tr[u].rs);
        }
    }

    int query(int l, int r, int vl, int vr, int k) {
        if (l == r) {
            return l;
        }
        int m = (l + r) >> 1;
        int lcnt = tr[tr[vr].ls].cnt - tr[tr[vl].ls].cnt;
        if (k <= lcnt) {
            return query(l, m, tr[vl].ls, tr[vr].ls, k);
        }
        return query(m + 1, r, tr[vl].rs, tr[vr].rs, k - lcnt);
    }
}

void solve() {
    int n, q;
    std::cin >> n >> q;
    for (int i = 1; i <= n; i++) {
        std::cin >> a[i];
        b[i] = a[i];
    }
    std::sort(b + 1, b + n + 1);
    int m = std::unique(b + 1, b + n + 1) - (b + 1);
    for (int i = 1; i <= n; i++) {
        int pos = std::lower_bound(b + 1, b + m + 1, a[i]) - b;
        PSGT::insert(1, m, root[i - 1], pos, root[i]);
    }
    while (q--) {
        int l, r, k;
        std::cin >> l >> r >> k;
        int p = PSGT::query(1, m, root[l - 1], root[r], k);
        std::cout << b[p] << '\n';
    }
}