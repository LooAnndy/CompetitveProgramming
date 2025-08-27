### 平衡树
```c++
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace std;
using namespace __gnu_pbds;
using pii = pair<int, int>;

using Tree = tree<pii, null_type, less<pii>, rb_tree_tag, tree_order_statistics_node_update>;
```
### 数据结构操作说明

| 操作 | 说明 |
|------|------|
| `pii : [val,order]` | 包含值(val)和顺序(order)，带order是为了避免value重复无法插入 |
| `下标从0开始` | 索引起始值说明 |
| `insert(x)` | 插入元素x |
| `erase(x)` | 删除元素x |
| `order_of_key(x)` | 返回严格小于x的元素个数 |
| `lower_bound(x)` | 返回大于等于x的第一个元素 |
| `upper_bound(x)` | 返回大于x的第一个元素 |
| `find_by_order(x)` | 返回第x小的元素（x为下标） |
| `x.split(k,y)` | 分割操作：将小于等于k的元素保留在x中，剩余元素移至y树 |