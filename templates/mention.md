## tricks and notes
### 中位数

对序列A二分x后的check

$$令f(a_i, x) = 
\begin{cases} 
1, & \text{若 } a_i \geq x, \\
-1, & \text{若 } a_i < x.
\end{cases}$$

如果总和$S=\sum f(a_i, x)>0$

右偏二分对x进行下取整取得答案

### 二维偏序

### note
前导0需要特判掉位数位1的情况，即数字'0'
