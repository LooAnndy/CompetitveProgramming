// 来源：https://www.cnblogs.com/EityDawn/articles/18534561
/*
用法：
支持全类型读入，用法 fin>>x;
支持全类型输出（__float128 与 long double 除外）用法 fout<<x;，对于字符数组指定可以输出指定位置到末尾的整个串，如 fout<<s+1。
对于字符数组，支持 fin>>s+1，c++20 以上版本并不会 CE。
实现了两个小功能
浮点数控制位数，用法 fout<<setdot(n);，调用后输出浮点数就会保留
 位小数，注意只有在这个语句后面的输出才会保留
 为小数，fout<<setdot(3)<<f<<' '<<setdot(6)<<f 前面会保留
 位，后面会保留
 位。
读入一整行字符串，用法 fin>>getline(s) 或者直接 getline(s)。
读入和输出是可以链式的，如 fin>>x>>y>>s+1>>f>>getline(t)>>... fout<<x<<' '<<y<<'\n'<<s+1<<setdot(3)<<f<<...。反正用法和 cin 与 cout 差不多就是了。
读入时要加 EOF 即 Ctrl+z 才会停下
注：字符串比较慢，不如iostream
*/
#include <bits/stdc++.h>
namespace FastIO {
    const int MAXN = (1 << 23);
    static char in[MAXN], *St = in, *Ed = in;
#define getchar() (St == Ed && (Ed = (St = in) + fread(in, 1, MAXN, stdin), St == Ed) ? EOF : *St++)
    static char out[MAXN], *now = out;
#define flush() (fwrite(out, 1, now - out, stdout))
#define putchar(x) (now == out + MAXN && (flush(), now = out), *now++ = (x))
    class Flush {
    public:
        ~Flush() { flush(); }
    } _;
    class In {
    public:
        template <typename T>
        inline In& operator>>(T& x) {
            x = 0;
            bool f = 0;
            char c = getchar();
            while (c < '0' || c > '9')
                f |= (c == '-'), c = getchar();
            while (c >= '0' && c <= '9')
                x = x * 10 + c - '0', c = getchar();
            if (c == '.') {
                c = getchar();
                double dot = 0.1;
                while (c >= '0' && c <= '9')
                    x += (c - '0') * dot, dot *= 0.1, c = getchar();
            }
            return (f ? x = -x : x), *this;
        }
        inline In& operator>>(char& x) {
            while (isspace(x = getchar()))
                ;
            return *this;
        }
        inline In& operator>>(char* x) {
            char c = getchar();
            while (isspace(c))
                c = getchar();
            while (!isspace(c) && ~c)
                *(x++) = c, c = getchar();
            return *x = 0, *this;
        }
        inline In& operator>>(std::string& x) {
            char c = getchar();
            x.clear();
            while (isspace(c))
                c = getchar();
            while (!isspace(c) && ~c)
                x.push_back(c), c = getchar();
            return *this;
        }
        inline In& getline(char* x) {
            char c = getchar();
            while (!(c == ' ' || !isspace(c)))
                c = getchar();
            while (c == ' ' || !isspace(c))
                (*x++) = c, c = getchar();
            return *x = 0, (*this);
        }
        inline In& getline(std::string& x) {
            char c = getchar();
            x.clear();
            while (!(c == ' ' || !isspace(c)))
                c = getchar();
            while (c == ' ' || !isspace(c))
                x.push_back(c), c = getchar();
            return (*this);
        }
        inline In& operator>>(In& in) { return in; }
    };
    class Out {
    private:
        char buf[20];
        short dot = 6, top = 0;

    public:
        template <typename T>
        inline Out& operator<<(T x) {
            if (x < 0) putchar('-'), x = -x;
            do
                buf[++top] = x % 10, x /= 10;
            while (x);
            while (top)
                putchar(buf[top--] ^ '0');
            return *this;
        }
        inline Out& operator<<(char c) { return putchar(c), *this; }
        inline Out& operator<<(std::string x) {
            for (auto c : x)
                putchar(c);
            return *this;
        }
        inline Out& operator<<(char* x) {
            while (*x)
                putchar(*(x++));
            return *this;
        }
        inline Out& operator<<(const char* x) {
            while (*x)
                putchar(*(x++));
            return *this;
        }
        inline Out& operator<<(double x) {
            snprintf(buf, sizeof(buf), "%.*lf", dot, x);
            return (*this) << buf;
        }
        inline Out& operator<<(Out& out) { return out; }
        inline Out setdot(const int n) { return dot = n, *this; }
    };
    static In fin;
    static Out fout;
    inline Out& setdot(const int n, Out& out = fout) { return fout.setdot(n), out; }
    inline In& getline(char* x, In& in = fin) { return fin.getline(x), in; }
    inline In& getline(std::string& x, In& in = fin) { return fin.getline(x), in; }
}
using namespace FastIO;