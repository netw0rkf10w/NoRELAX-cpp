#include "lemmas.hpp"
#include "projection.h"

void SimplexProjection(Eigen::VectorXd &x, const Eigen::VectorXd &c)
/// Solve   min     1/2*||x||^2 - c^T*x
///         subject to      1^T*x = 1 and x >= 0.
/// Input c: n*1 vector
/// Output x: n*1 vector
{
    const double* y = c.data();
    unsigned int n = c.size();
    double *z = new double[n];
    simplexproj_Condat(y, z, n, 1.0);
    //simplexproj_Duchi(y, z, n, 1.0);
    x = Eigen::Map<Eigen::VectorXd>(z, n); // no need to free z
    delete z;
}


void SimplexProjection_INEQUALITY(Eigen::VectorXd &x, const Eigen::VectorXd &c)
/// Solve   min     1/2*||x||^2 - c^T*x
///         subject to      1^T*x <= 1 and x >= 0.
/// Input c: n*1 vector
/// Output x: n*1 vector
{
//    x = Eigen::VectorXd::Zero(c.size());
//    double s = 0.0;
//    for(size_t i = 0; i < x.size(); i++)
//    {
//        if(c(i) > 0)
//        {
//            x(i)= c(i);
//            s += c(i);
//        }
//    }
//    if(s <= 1)
//        return;

    x = c.cwiseMax(0.0);
    double  s = x.sum();
    if(s <= 1)
        return;
    SimplexProjection(x, c);
}


void SimplexProjection(Eigen::MatrixXd &X, const Eigen::MatrixXd &C)
/// Project each column C.col(i) onto the simplex
/// The result is stored in X.col(i)
/// Input C: L*V matrix
/// Output X: L*V matrix
{
    X = Eigen::MatrixXd::Zero(C.rows(), C.cols());
    unsigned int n = C.rows();
    #pragma omp parallel for
    for(size_t i = 0; i < (size_t)C.cols(); i++){
        Eigen::VectorXd cc = C.col(i);
        const double* y = cc.data();
        double *z = new double[n];
        simplexproj_Condat(y, z, n, 1.0);
        //simplexproj_Duchi(y, z, n, 1.0);
        X.col(i) = Eigen::Map<Eigen::VectorXd>(z, n);
        delete z;
    }
}
