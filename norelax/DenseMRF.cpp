#include <iomanip>
#include "DenseMRF.hpp"
#include "lemmas.hpp"

using namespace std;
using namespace Eigen;

DenseMRF::DenseMRF()
{

}


double DenseMRF::ADMM()
{
    if(parameters.verbose){
        cout<<"**********************************************************"<<endl;
        cout<<"Number of nodes: "<<numberOfNodes<<endl;
        //cout<<"Number of edges: "<<numberOfEdges<<endl;
        cout<<"ADMM parameters:"<<endl;
        cout<<"rho_min = "<<parameters.rho_min<<endl<<"rho_max = "<<parameters.rho_max<<endl<<"step = "<<parameters.step<<endl<<"maxIt = "<<parameters.MAX_ITER<<endl;
        cout<<"iter1 = "<<parameters.iter1<<endl<<"iter2 = "<<parameters.iter2<<endl<<"protocol = "<<parameters.protocolFrequence<<endl<<"precision = "<<parameters.precision<<endl;
        cout<<"timeout = "<<parameters.timeout<<endl;
        cout<<"**********************************************************"<<endl;
    }


    // Instead of normalizing the potentials to [-1,1] by dividing them by max_potential_value
    // (and storing the normalized potentials) we can save the memory by scaling rho (and scaling
    // the initial dual variables y as well, unless it is set to 0)
    /// Instead of normalizing the potentials to [-1,1], we equivalently scale
    /// the penalty parameter rho and the initial dual variable Y
//    double absmax = nodePotentials.cwiseAbs().maxCoeff();
//    for(size_t e = 0; e < numberOfEdges; e++){
//        double cmax = edgePotentials[e].cwiseAbs().maxCoeff();
//        if(cmax > absmax)
//            absmax = cmax;
//    }

    double rho_min = absMax*parameters.rho_min;
    double rho_max = absMax*parameters.rho_max;


    // Output

    size_t V = numberOfNodes;
    size_t L = numberOfLabels[0];
    size_t A = V*L;

    //Eigen::VectorXd x1 = Eigen::VectorXd::Ones(A)/std::max(n1,n2);
    Eigen::VectorXd x1 = Eigen::VectorXd::Map(X.data(), A);
    Eigen::VectorXd x2 = x1;
    Eigen::VectorXd y = Eigen::VectorXd::Zero(A);


    double energy;
    double energy_best = DBL_MAX;
    double rho = rho_min;
    int iter1_cumulated = parameters.iter1;
    double res_best_so_far = DBL_MAX;
    int counter = 0;
    double residual = DBL_MAX, residual_old;

    myclock::time_point begin, end;
    myclock::time_point begin2, end2;
    double T1 = 0.0, T2 = 0.0;


    double total_time;
    if(visitor.times.empty())
        total_time = 0.0;
    else
        total_time = visitor.times.back();

    size_t last_iter;
    if(visitor.iteration.empty())
        last_iter = 0;
    else
        last_iter = visitor.iteration.back();

    begin = myclock::now();

    double r, s;
    for(int k = 0; k <= parameters.MAX_ITER; k++)
    {
        /// Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        /// c1 = x2 - (d + M*x2 + y)/rho

        begin2 = myclock::now();
        Eigen::VectorXd c1 = x2 - (d + M*x2 + y)/rho;
        Eigen::Map<Eigen::MatrixXd> C1(c1.data(), L, V);
        end2= myclock::now();
        T1 += measure_time(begin2, end2);

        begin2 = myclock::now();
        Eigen::VectorXd x1_old = x1;
        Eigen::MatrixXd X1;
        SimplexProjection(X1, C1);
        x1 = Eigen::VectorXd::Map(X1.data(), A);
        end2 = myclock::now();
        T2 += measure_time(begin2, end2);

        /// Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
        /// c2 = x1 - (M^T*x1 - y)/rho
        begin2 = myclock::now();
        Eigen::VectorXd c2 = x1 + (y - M*x1)/rho;
        Eigen::Map<Eigen::MatrixXd> C2(c2.data(), L, V);
        end2= myclock::now();
        T1 += measure_time(begin2, end2);

        begin2 = myclock::now();
        Eigen::VectorXd x2_old = x2;
        Eigen::MatrixXd X2 = C2.cwiseMax(0.0);
        x2 = Eigen::VectorXd::Map(X2.data(), A);
        end2 = myclock::now();
        T2 += measure_time(begin2, end2);


        /// Step 3: update y
        y += rho*(x1 - x2);

        /// Step 4: compute the residuals and update rho

        if(k%parameters.protocolFrequence == 0 || k >= parameters.MAX_ITER - 1){
            end = myclock::now();
            total_time += measure_time(begin, end);
            visitor.times.push_back(total_time);

            visitor.iteration.push_back(k + last_iter + 1);

            /// Step 4: compute the residuals and update rho
            r = (x1 - x2).squaredNorm();
            s = (x1 - x1_old).squaredNorm() + (x2 - x2_old).squaredNorm();

            residual_old = residual;
            residual = r + s;

            //energy =  this->energy(XD[0]);
            energy = x1.dot(d) + x1.dot(M*x1);

            visitor.energies.push_back(energy);
            visitor.residuals.push_back(residual);
            //visitor.bounds.push_back(-DBL_MAX);

            if(energy < energy_best){
                energy_best = energy;
                X = X1;
            }

            if(parameters.verbose)
                cout<<k+1<<")\tresidual: "<<setw(10)<<residual<<"\tenergy: "<<setw(10)<<energy<<endl;

            /// If convergence
            if(residual <= parameters.precision && residual_old <= parameters.precision)
                break;


            /// If timeout reached
            if(total_time > parameters.timeout){
                if(parameters.verbose){
                    cout<<"--- TIMEOUT ---"<<endl;
                }
                break;
            }

            begin = myclock::now();
        }

        /// Only after iter1 iterations that we start to track the best residuals
        if(k >= iter1_cumulated){
            if(residual < res_best_so_far){
                res_best_so_far = residual;
                counter = 0;
            }else{
                counter++;
            }
            // If the best_so_far residual has not changed during iter2 iterations, then update rho
            if(counter >= parameters.iter2){
                if(rho < rho_max){
                    rho = std::min(rho*parameters.step, rho_max);
                    if(parameters.verbose){
                        cout<<k+1<<") --- UPDATE rho = "<<rho/absMax<<endl;
                    }
                    counter = 0;
                    iter1_cumulated = k + parameters.iter1;
                }else{
                    break;
                }
            }
        }
    }


    /// If having reached the maxium number of iterations without convergence,
    /// then return the current solution and the residuals


    if(parameters.verbose){
        cout<<"Coefficient computation time:\t "<<T1<<endl;
        cout<<"Simplex projection time:\t "<<T2<<endl;
        cout<<"Total ADMM time:\t "<<total_time<<endl;
    }

    // Convert to labels
    states.resize(V);
    for(size_t i = 0; i < V; i++){
        size_t label_max;
        X.col(i).maxCoeff( &label_max );
        //if(x_max < 1.0) cout<<x_max<<", ";
        states[i] = label_max;
    }

    return energy_best;
}
