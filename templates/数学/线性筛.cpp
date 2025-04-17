#include <bits/stdc++.h>
const int N = 1E6 + 10;

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

int main() {
    euler(n);
}