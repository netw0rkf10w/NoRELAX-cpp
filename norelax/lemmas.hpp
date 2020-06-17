#ifndef LEMMAS_H
#define LEMMAS_H

#include <omp.h>
#include <Eigen/Dense>

void SimplexProjection(Eigen::VectorXd &x, const Eigen::VectorXd &c);
/// Solve   min     1/2*||x||^2 - c^T*x
///         subject to      1^T*x = 1 and x >= 0.
/// Input c: n*1 vector
/// Output x: n*1 vector


void SimplexProjection_INEQUALITY(Eigen::VectorXd &x, const Eigen::VectorXd &c);
/// Solve   min     1/2*||x||^2 - c^T*x
///         subject to      1^T*x <= 1 and x >= 0.
/// Input c: n*1 vector
/// Output x: n*1 vector


void SimplexProjection(Eigen::MatrixXd &X, const Eigen::MatrixXd &C);
/// Project each column C.col(i) onto the simplex
/// The result is stored in X.col(i)
/// Input C: L*V matrix
/// Output X: L*V matrix

#endif // LEMMAS_H
