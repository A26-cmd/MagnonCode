#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

//ここのハミルトニあんは定数こうを引いていないことに注意
using namespace std;
using namespace Eigen;

// j番目のビット（スピン）が立っているか（下向きか）を判定する関数
bool check_bit(int mask, int j) {
    return (mask & (1 << j)) != 0;
}

int main() {
    // 1. 基底の構築 (6サイト中、2サイトが下向きのスピン状態を探す)
    vector<int> basis;
    for (int i = 0; i < (1 << 6); ++i) {
        int down_spins = 0;
        for (int j = 0; j < 6; ++j) {
            if (check_bit(i, j)) down_spins++;
        }
        if (down_spins == 2) {
            basis.push_back(i);
        }
    }
    int dim = basis.size(); // 6C2 = 15次元のヒルベルト空間
    
    // 2. ヘキサゴン内のハミルトニアン行列 H_hex の構築
    MatrixXd H = MatrixXd::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) {
        int state = basis[i];
        double sz_sz = 0;
        for (int site = 0; site < 6; ++site) {
            int next_site = (site + 1) % 6;
            bool s1 = check_bit(state, site);
            bool s2 = check_bit(state, next_site);
            
            // 対角成分: Sz Sz
            if (s1 == s2) {
                sz_sz += 0.25; // 平行なら +1/4
            } else {
                sz_sz -= 0.25; // 反平行なら -1/4
                
                // 非対角成分: S+ S- + S- S+ (反平行スピンをフリップ)
                int flipped_state = state ^ (1 << site) ^ (1 << next_site);
                int flipped_idx = -1;
                for (int j = 0; j < dim; ++j) {
                    if (basis[j] == flipped_state) {
                        flipped_idx = j;
                        break;
                    }
                }
                H(flipped_idx, i) += 0.5; // 交換相互作用の係数 1/2
            }
        }
        H(i, i) += sz_sz;
    }
    
    // 3. 論文の (8) 式の変分状態ベクトルを構築
    VectorXd psi = VectorXd::Zero(dim);
    double sqrt5 = sqrt(5.0);
    
    // 運動量 k=0 の規格化係数 (Class A, B は 6状態、Class C は 3状態の重ね合わせ)
    double c_A = (sqrt5 - 1.0) / (2.0 * sqrt5 * sqrt(6.0));
    double c_B = -(sqrt5 + 1.0) / (2.0 * sqrt5 * sqrt(6.0));
    double c_C = (2.0 * sqrt(2.0)) / (2.0 * sqrt5 * sqrt(3.0));
    
    for (int i = 0; i < dim; ++i) {
        int state = basis[i];
        int pos1 = -1, pos2 = -1;
        for (int j = 0; j < 6; ++j) {
            if (check_bit(state, j)) {
                if (pos1 == -1) pos1 = j;
                else pos2 = j;
            }
        }
        // リング上での2つの下向きスピンの最短距離
        int dist = min(pos2 - pos1, 6 - (pos2 - pos1));
        
        if (dist == 1) {
            psi(i) = c_A; // 隣接 (Class A)
        } else if (dist == 2) {
            psi(i) = c_B; // 1つ飛ばし (Class B)
        } else if (dist == 3) {
            psi(i) = c_C; // 向かい合わせ (Class C)
        }
    }
    
    // 4. ヘキサゴン内でのエネルギー期待値 E_hex
    double E_hex = psi.transpose() * H * psi;
    cout << "Energy of isolated hexagon: E_hex = " << E_hex << endl;
    
    // 5. 格子全体の変分エネルギーの計算
    // 9サイト単位胞において、外部の3つの上向きスピン(S=1/2)との結合(12本)のエネルギー
    // 各ヘキサゴンサイトは外部スピンと2本繋がるため: sum_i S^z_i * (2 * 1/2) = S^z_total_hex
    // 4つの上向き、2つの下向きなので S^z_total_hex = 4*(0.5) + 2*(-0.5) = 1.0
    double E_inter = 1.0; 
    
    double E_unit = E_hex + E_inter;
    double e_variational = E_unit / 9.0; // 1サイトあたりのエネルギー
    
    cout << "Variational energy per site (calculated): e = " << e_variational << endl;
    cout << "Paper's value: -sqrt(5)/18 = " << -sqrt5 / 18.0 << endl;
    
    return 0;
}