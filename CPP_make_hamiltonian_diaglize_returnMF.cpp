#include <algorithm>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

//コンパイルする時はlib_make_hamiltonian.soでコンパイルして、julia側でusingして呼び出す感じで

using namespace std;

int MF_size = 18;//6つの3成分ある平均場
int ndim = pow(2,9);

int nsign(size_t k, int n) {
  k = (k >> n) & 1;
  return 1 - 2 * k;
}
size_t reverse(size_t k, int n) {
  int l = 1;
  l = (l << n);
  k = k ^ l;
  return k;
}

auto op_xx = [](size_t &k, int i, int j) -> double {
  if (i == j) {
    return 1.;
  } else {
    k = reverse(k, i);
    k = reverse(k, j);
    return 1.;
  }
};

auto op_yy = [](size_t &k, int i, int j) -> double {
  if (i == j) {
    return 1.;
  } else {
    k = reverse(k, i);
    k = reverse(k, j);
    return -1. * (double)(nsign(k, i) * nsign(k, j));
  }
};

auto op_zz = [](size_t &k, int i, int j) -> double {
  if (i == j) {
    return 1.;
  } else {
    return 1. * (double)(nsign(k, i) * nsign(k, j));
  }
};

auto op_x = [](size_t &k, int i) -> double {
  k = reverse(k, i);
  return 1.0;
};
auto op_y = [](size_t &k, int i) -> complex<double> {
  k = reverse(k, i);
  return complex<double>(0, -1) * (double)(nsign(k, i));
};

auto op_z(size_t &k, int i) -> double { return (double)(nsign(k, i)); };

template <typename T>
inline void xx_op(T const &c1, T &c2, int i, int j, double c) {
  for (size_t n = 0; n < c1.size(); n++) {
    size_t k = n;
    auto val = op_xx(k, i, j);
    c2[n] += c * val * c1[k];
  }
}
template <typename T>
inline void yy_op(T const &c1, T &c2, int i, int j, double c) {
  for (size_t n = 0; n < c1.size(); n++) {
    size_t k = n;
    auto val = op_yy(k, i, j);
    c2[n] += c * val * c1[k];
  }
}
template <typename T>
inline void zz_op(T const &c1, T &c2, int i, int j, double c) {
  for (size_t n = 0; n< c1.size(); n++) {
    size_t k = n;
    auto val = op_zz(k, i, j);
    c2[n] += c * val * c1[k];
  }
}

template <typename T, typename U>
inline void z_op(T const &c1, U &c2, int i, double c) {
  for (size_t n = 0; n < c1.size(); n++) {
    size_t k = n;
    auto val = op_z(k, i);
    c2[n] += c * val * c1[k];
  }
}
template <typename T, typename U>
inline void y_op(T const &c1, U &c2, int i, double c) {
  for (size_t n = 0; n < c1.size(); n++) {
    size_t k = n;
    auto val = op_y(k, i);
    c2[n] += c * val * c1[k];
  }
}
template <typename T, typename U>
inline void x_op(T const &c1, U &c2, int i, double c) {
  for (size_t n = 0; n < c1.size(); n++) {
    size_t k = n;
    auto val = op_x(k, i);
    c2[n] += c * val * c1[k];
  }
}

//なんかz_opがEigenのベクトルだと動いてくれないので、オーバーロード
#include <Eigen/Dense>

using type = double;


// 使用する行列の型を指定
using Mat = Eigen::Matrix<complex<double>, Eigen::Dynamic, Eigen::Dynamic>;
using Vtype = Eigen::Matrix<type, Eigen::Dynamic, 1>;
using Vctype = Eigen::Matrix<complex<double>, Eigen::Dynamic, 1>;
//MFの与え方：<S1245>_x,<S_1245>_y,<S1245>_z,<S1346>_x,<S1346>_y,<S1346>_z,<S2356>_x,<S2356>_y,<S2356>_z,<SA>_x,<SA>_y, <SA>_z,<SB>_x,<SB>_y,<SB>_z,<SC>_x,<SC>_y,<SC>_zの順
//c1,c2:0~5サイトは研究ノートの通り、6,7,8がA,B,C
//MFを基底状態に対して計算してくれる。

extern "C" {
    void build_and_solve_mf_hamiltonian(double J, double h, double* MF_data ,double* MFout, complex<double>* out_ptr){
        std::vector<double> MF(MF_data, MF_data + MF_size);
        //cout<<"test"<<endl;
        auto hamil = [J, h, MF](const Vctype &c1, Vctype &c2) {

                //六角形部分
                for(int i=0; i<5;i++){
                    xx_op(c1,c2, i,i+1, J);
                    yy_op(c1,c2, i,i+1, J);
                    zz_op(c1,c2, i,i+1, J);
                }
                xx_op(c1,c2,0,5,J);
                yy_op(c1,c2,0,5,J);
                zz_op(c1,c2,0,5,J);

                //平均場＋磁場部分
                x_op(c1,c2,6,MF[0]*J);
                y_op(c1,c2,6,MF[1]*J);
                z_op(c1,c2,6,MF[2]*J-h);
                x_op(c1,c2,7,MF[3]*J);
                y_op(c1,c2,7,MF[4]*J);
                z_op(c1,c2,7,MF[5]*J-h);
                x_op(c1,c2,8,MF[6]*J);
                y_op(c1,c2,8,MF[7]*J);
                z_op(c1,c2,8,MF[8]*J-h);
                
                x_op(c1,c2,0,(MF[9]+MF[12])*J);
                y_op(c1,c2,0,(MF[10]+MF[13])*J);
                z_op(c1,c2,0,(MF[11]+MF[14])*J-h);
                x_op(c1,c2,1,(MF[9]+MF[15])*J);
                y_op(c1,c2,1,(MF[10]+MF[16])*J);
                z_op(c1,c2,1,(MF[11]+MF[17])*J-h);
                x_op(c1,c2,2,(MF[12]+MF[15])*J);
                y_op(c1,c2,2,(MF[13]+MF[16])*J);
                z_op(c1,c2,2,(MF[14]+MF[17])*J-h);
                x_op(c1,c2,3,(MF[9]+MF[12])*J);
                y_op(c1,c2,3,(MF[10]+MF[13])*J);
                z_op(c1,c2,3,(MF[11]+MF[14])*J-h);
                x_op(c1,c2,4,(MF[9]+MF[15])*J);
                y_op(c1,c2,4,(MF[10]+MF[16])*J);
                z_op(c1,c2,4,(MF[11]+MF[17])*J-h);
                x_op(c1,c2,5,(MF[12]+MF[15])*J);
                y_op(c1,c2,5,(MF[13]+MF[16])*J);
                z_op(c1,c2,5,(MF[14]+MF[17])*J-h);

        };

        Mat H = Mat::Zero(ndim, ndim);
        for (size_t j = 0; j < ndim; ++j) {
            Vctype c1(ndim);
            c1.setZero();
            c1[j] = 1.0;
            Vctype c2(ndim);
            c2.setZero();
            hamil(c1, c2);
            for (size_t i = 0; i < ndim; ++i) H(i, j) = c2[i];
        }
        // Eigenを使った対角化
        Eigen::SelfAdjointEigenSolver<Mat> es(H);
        auto w = es.eigenvalues();
        auto v = es.eigenvectors();

        //基底状態による新しい平均場の計算
        Vctype gs=v.col(0);
        vector<double> tmp(18);
        
        Vctype c2 = Vctype::Zero(ndim);
        
        x_op(gs, c2, 0, 1.0);
        tmp[0] = (gs.adjoint() * c2).value().real();
        
        c2.setZero();
        y_op(gs, c2, 0, 1.0);
        tmp[1] = (gs.adjoint() * c2).value().real();
        
        c2.setZero();
        z_op(gs, c2, 0, 1.0);
        tmp[2] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        x_op(gs, c2, 1, 1.0);
        tmp[3] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        y_op(gs, c2, 1, 1.0);
        tmp[4] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        z_op(gs, c2, 1, 1.0);
        tmp[5] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        x_op(gs, c2, 2, 1.0);
        tmp[6] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        y_op(gs, c2, 2, 1.0);
        tmp[7] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        z_op(gs, c2, 2, 1.0);
        tmp[8] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        x_op(gs, c2, 3,1.0);
        tmp[9] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        y_op(gs, c2, 3, 1.0);
        tmp[10] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        z_op(gs, c2, 3, 1.0);
        tmp[11] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        x_op(gs, c2, 4, 1.0);
        tmp[12] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        y_op(gs, c2, 4, 1.0);
        tmp[13] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        z_op(gs, c2, 4, 1.0);
        tmp[14] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        x_op(gs, c2, 5, 1.0);
        tmp[15] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        y_op(gs, c2, 5, 1.0);
        tmp[16] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        z_op(gs, c2, 5, 1.0);
        tmp[17] = (gs.adjoint() * c2).value().real();
        c2.setZero();
        MFout[0]=tmp[0]+tmp[3]+tmp[9]+tmp[12];
        MFout[1]=tmp[1]+tmp[4]+tmp[10]+tmp[13];
        MFout[2]=tmp[2]+tmp[5]+tmp[11]+tmp[14];
        MFout[3]=tmp[0]+tmp[6]+tmp[9]+tmp[15];
        MFout[4]=tmp[1]+tmp[7]+tmp[10]+tmp[16];
        MFout[5]=tmp[2]+tmp[8]+tmp[11]+tmp[17];
        MFout[6]=tmp[3]+tmp[6]+tmp[12]+tmp[15];
        MFout[7]=tmp[4]+tmp[7]+tmp[13]+tmp[16];
        MFout[8]=tmp[5]+tmp[8]+tmp[14]+tmp[17];

        x_op(gs,c2,6,1.0);
        MFout[9]= (gs.adjoint() * c2).value().real();
        c2.setZero();
        y_op(gs,c2,6,1.0);
        MFout[10]= (gs.adjoint() * c2).value().real();
        c2.setZero();
        z_op(gs,c2,6,1.0);
        MFout[11]= (gs.adjoint() * c2).value().real();
        c2.setZero();
        x_op(gs,c2,7,1.0);
        MFout[12]= (gs.adjoint() * c2).value().real();
        c2.setZero();
        y_op(gs,c2,7,1.0);
        MFout[13]= (gs.adjoint() * c2).value().real();
        c2.setZero();
        z_op(gs,c2,7,1.0);
        MFout[14]= (gs.adjoint() * c2).value().real();
        c2.setZero();
        x_op(gs,c2,8,1.0);
        MFout[15]= (gs.adjoint() * c2).value().real();
        c2.setZero();
        y_op(gs,c2,8,1.0);
        MFout[16]= (gs.adjoint() * c2).value().real();
        c2.setZero();
        z_op(gs,c2,8,1.0);
        MFout[17]= (gs.adjoint() * c2).value().real();



        //ここ不安なのでjuliaで動作を確認したい。
        // out_ptr に固有値・固有ベクトルを詰める (Julia側で (ndim + 1) * ndim 分の領域を確保しておく)
        // 1. 固有値を最初の ndim 個にコピー
        for (int i = 0; i < ndim; ++i) {
            out_ptr[i] = w(i);
        }

        // 2. 固有ベクトルをその後にコピー (列優先で並ぶ)
        // インデックス ndim から開始
        for (int i = 0; i < ndim * ndim; ++i) {
            out_ptr[ndim + i] = v.data()[i];
        }
    }
}
