### 光速幂(k进制倍增)
对一个数a考虑分块预处理，令 $r = \lfloor \sqrt{p} \rfloor $，则我们可以把 $ a^x $ 分成几个整块的 $a^r$ 和一个散的$a_q$,

预处理 
$$a^r, (a^r)^2, (a^r)^3, \cdots, (a^r)^r, a, a^2 \cdots, a^r $$

然后我们就可以求 $a^x$ 为  

$$ a_x = a^{rp + q} = (a^r)^p \cdot a^q $$  

这里 $ rp + q $ 类似一个带余除法 $ 0 \leq q < r $。  

感觉矩阵使用的多一些

### 最大公约数

