#include <bits/stdc++.h>
using i64 = long long;
constexpr i64 Mod = 998244353;

struct Mat {
    static constexpr int M = 32;
    i64 mat[M][M];

    Mat() { memset(mat, 0, sizeof mat); }

    static Mat identity() {
        Mat res;
        for (int i = 0; i < M; i++) {
            res.mat[i][i] = 1;
        }
        return res;
    }

    i64* operator[](int i) { return mat[i]; }
    const i64* operator[](int i) const { return mat[i]; }

    Mat operator*(const Mat& B) {
        Mat A = *this;
        Mat res;
        for (int k = 0; k < M; k++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    res[i][j] = (res[i][j] + A[i][k] * B[k][j]) % Mod;
                }
            }
        }
        return res;
    }

    Mat operator+(const Mat& B) {
        Mat A = *this;
        Mat res;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                res[i][j] = (A[i][j] + B[i][j]) % Mod;
            }
        }
        return res;
    }

    Mat pow(i64 b) const {
        Mat res = Mat::identity();
        Mat a = *this;

        for (; b; b >>= 1) {
            if (b & 1) res = res * a;
            a = a * a;
        }
        return res;
    }

    friend std::ostream& operator<<(std::ostream& os, const Mat& m) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                os << m.mat[i][j] << " ";
            }
            os << "\n";
        }
        return os;
    }
};
