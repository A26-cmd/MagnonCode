#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main() {
    const int N = 9;                  // サイト数
    const int dim = pow(2, N);        // 512次元
    const double J = 1.0;             // 結合定数

    // 512x512の行列をゼロ初期化
    MatrixXd ham = MatrixXd::Zero(dim, dim);

    cout << "Building Hamiltonian for N=" << N << " (Dim=" << dim << ")..." << endl;

    // ハミルトニアンの構成: H = J * sum(Si * Si+1)
    for (int i = 0; i < dim; ++i) {
        for (int site = 0; site < N; ++site) {
            int next_site = (site + 1) % N; // 周期境界条件

            // 状態 i の各ビットを確認
            int bit_i = (i >> site) & 1;
            int bit_j = (i >> next_site) & 1;

            // Sz * Sz 項 (対角項)
            // bitが同じなら +1/4, 違えば -1/4
            if (bit_i == bit_j) ham(i, i) += 0.25 * J;
            else                ham(i, i) -= 0.25 * J;

            // スピン交換項 (S+S- + S-S+)/2 (非対角項)
            // 隣接するビットが異なる場合のみ、入れ替えた状態へ遷移
            if (bit_i != bit_j) {
                // bitを反転させるマスクを作成
                int flip_mask = (1 << site) | (1 << next_site);
                int j = i ^ flip_mask; 
                ham(i, j) += 0.5 * J;
            }
        }
    }

    cout << "Diagonalizing..." << endl;

    // 対角化の実行
    SelfAdjointEigenSolver<MatrixXd> solver(ham);

    if (solver.info() != Success) {
        cerr << "Diagonalization failed!" << endl;
        return 1;
    }

// 結果表示
    cout << "Done!" << endl;
    
    // 全固有値を取得
    VectorXd energies = solver.eigenvalues();

    cout << "--- All Eigenvalues (Total: " << energies.size() << ") ---" << endl;
    for (int i = 0; i < energies.size(); ++i) {
        // インデックス、エネルギー、(もしあれば)1つ前との差（ギャップ）を表示
        printf("[%3d] %15.10f", i, energies(i));
        
        if (i > 0) {
            printf("  (gap: %15.10f)", energies(i) - energies(i-1));
        }
        printf("\n");
    }

    cout << "------------------------------------------" << endl;
    cout << "Ground state energy: " << energies(0) << endl;

    return 0;
}