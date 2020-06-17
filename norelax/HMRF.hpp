#ifndef HMRF_H
#define HMRF_H

#include <cfloat>
#include <chrono>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <set>

#include <omp.h>

#include "OpenGM.hpp"
#include "outils_io.hpp"

typedef std::vector<int> VI;
typedef std::vector<VI> VVI;
typedef std::vector<size_t> VS;
typedef std::vector<VS> VVS;
typedef std::vector<double> VD;
typedef std::vector<VD> VVD;


typedef std::chrono::steady_clock myclock;
double measure_time(myclock::time_point begin, myclock::time_point end);
void getTensorIndices_Loop(VVS &TensorIndices, VS &currentIndex, const size_t depth, const size_t L, const size_t S);
void getTensorIndices(VVS &TensorIndices, const size_t L, const size_t S);
void TensorContraction(Eigen::VectorXd &p, const VS &factor, const VD &potential, const VVS &LabelIndices, const std::vector<Eigen::MatrixXd> &X, const size_t d);
void TensorContraction(Eigen::VectorXd &p, const VS &factor, const VD &potential, const VVS &TensorIndices, const Eigen::MatrixXd &X, const size_t d);

#define NONE 0
#define EQUALITY 1
#define INEQUALITY 2
#define ROW 1
#define COL 2
#define VARIANT_1 1
#define VARIANT_2 2

#define MRF_TYPE_HIGHORDER 0
#define MRF_TYPE_PAIRWISE 1
#define MRF_TYPE_PAIRWISE_METRIC 2

#define X_TYPE_DENSE 0
#define X_TYPE_SPARSE 1
#define X_TYPE_DISCRETE 2


// V: number of nodes, L: number of labels, E: number of edges, H: number of hyper-edges

class HMRF
{
public:
    struct Parameters{
        // Common parameters
        int MAX_ITER = 20000;
        bool verbose = false;
        int protocolFrequence = 100;
        double precision = 1e-5;
        double timeout = 3600;

        // ADMM parameters
        double rho_min = 0.01;
        double rho_max = 100.0;
        double step = 1.2;
        int iter1 = 500;
        int iter2 = 300;

        // PDG or FW parameters
        bool lineSearch = false;
        int numInit = 1;
    };

    struct Visitor{
        VS iteration;
        VD energies;
        VD bounds;
        VD residuals;
        VD times;
    };

    Visitor visitor;
private:
    size_t numberOfFactors;                 // the number of factors (INCLUDING SINGLE NODES)
    VVS factors;                            // factors[c] contains the nodes of the c-th factor
    VVS containing_factors;                 // containing_factors[i] is the list of the factors that contain the node i
    VVD potentials;
    std::vector<VVS> TensorIndicesOfAllRanks;
protected:
    size_t max_degree;                      // maximum size of the factors
    size_t numberOfNodes;                   // shortly denoted by V in the comments
    VS numberOfLabels;                      // numberOfLabels[i] is the number of labels of node i
    VS states;                              // optimal solution
    Parameters parameters;
    std::string output = "output.h5";
    Eigen::MatrixXd X; // The continuous assignment matrix
    std::vector< std::set<size_t> > neighbors; // neighbors[i] is the set of neighbors of the node 'i' (two are neighbors if they belong to the same clique)
    double absMax; // Maxium absolute value of all potentials
public:
    HMRF();
    virtual ~HMRF(){}

    /*
    void setNumberOfLabels(const VS &_numberOfLabels){ numberOfLabels = _numberOfLabels; }
    void setNumberOfNodes(size_t V){ numberOfNodes = V; }
    void setNumberOfFactors(size_t F){ numberOfFactors = F; }
    void setFactors(const VVS &_factors){ factors = _factors; }
    void setNodes(const VVS &_nodes){ nodes = _nodes; }
    void setStates(const VS &_states){ states = _states; }
    */



    VS getNumberOfLabels(){ return numberOfLabels; }
    size_t getNumberOfNodes(){ return numberOfNodes; }
    size_t getNumberOfFactors(){ return numberOfFactors; }
    VVS getFactors(){ return factors; }
    VVS getNodes(){ return containing_factors; }
    VS getStates(){ return states; }
    VVD getPotentials(){ return potentials; }

    void setX(const Eigen::MatrixXd &XX){ X = XX; }
    Eigen::MatrixXd getX(){ return X; }
    void setParameters(const Parameters &param){ parameters = param; }
    void setOutputLocation(const std::string &outputfile){ output = outputfile; }

    /// Model importation
    template <typename GraphicalModelType>
    void importModel(const GraphicalModelType &gm);

    //void importStereoModel(float sigma, float smoothScale, const std::string &proposalsFile, const std::string &unariesFile, const std::string &horizontalWeightsFile, const std::string &verticalWeightsFile);

    void saveResults(const std::string &outputfile);
private:
    virtual void computeP(size_t d, const std::vector<Eigen::MatrixXd> &XD);
    std::vector<Eigen::MatrixXd> RC;
    std::vector< std::vector<bool> > hasChangedRC;
    std::vector<Eigen::MatrixXd> P;
    std::vector< std::vector<bool> > hasChangedP;
public:

    virtual double energy();
    virtual double energy(const Eigen::MatrixXd &_X);

    virtual double energy_discrete();


    /// Inference components (not all methods are used in the full pipeline)
    virtual void inference(const std::string &method = "admm");   // Full inference pipeline
    virtual double ADMM();        // ADMM
    virtual double ADMM_old();        // ADMM
    virtual double BCD(int numIter = 1000, bool random = true, int type = X_TYPE_DENSE);         // Block coordinate descent
    //void BCD_backup(bool random = true);
    //virtual void BCD_slow(bool random = true);         // Block coordinate descent
    virtual double PGD();         // Projected gradient descent
    virtual double FW();          // Frank–Wolfe algorithm (a.k.a Conditional gradient descent)
    virtual double CQP();    // Convex quadratic programming relaxation
    virtual void lineSearch(double &alpha_best, double &p_best, double energy, const Eigen::MatrixXd &R);

    // Compute the gradient of the energy
    virtual void gradient(Eigen::MatrixXd &G, int type = X_TYPE_DENSE);
    //virtual void gradient_discrete(Eigen::MatrixXd &G);
    virtual void gradient2(Eigen::MatrixXd &G); // Second version, for debugging
    // The partial derivative over node i only
    virtual void gradient(Eigen::MatrixXd &G, size_t i, int type = X_TYPE_DENSE);
    //virtual void gradient_discrete(Eigen::MatrixXd &G, size_t i);

    virtual void getUnarySolution(Eigen::MatrixXd &_X);
};


template <typename GraphicalModelType>
void HMRF::importModel(const GraphicalModelType &gm)
/// numLabels[i]: the number of labels of the node i
/// factors
{
    numberOfNodes = gm.numberOfVariables();
    numberOfLabels.resize(numberOfNodes);
    for(size_t i = 0; i < numberOfNodes; i++){
        numberOfLabels[i] = gm.numberOfLabels(i);
    }
    // In our model, a single node IS considered as a factor
    numberOfFactors = gm.numberOfFactors();

    /// Compute the graph structures
    /// 1. The list of nodes in each factor
    /// 2. The list of factors that each node belongs to
    /// 3. The number of factors of each degree (this is pre-computed for later efficency)

    factors.clear();
    potentials.clear();
    containing_factors.clear();
    factors.reserve(numberOfFactors);
    potentials.reserve(numberOfFactors);
    containing_factors.resize(numberOfNodes);
    max_degree = 0;



    for(size_t c = 0; c < numberOfFactors; ++c){
        size_t S = gm[c].numberOfVariables();
        assert(S > 0);
        // Add the factor nodes
        VS factor;
        factor.reserve(S);
        for(size_t idx = 0; idx < S; ++idx){
            size_t i = gm[c].variableIndex(idx);
            factor.push_back(i);
            containing_factors[i].push_back(c);
        }
        factors.push_back(factor);
        // Add the corresponding potentials
        VD potential(gm[c].size());
        gm[c].copyValuesSwitchedOrder(potential.begin());
        potentials.push_back(potential);

        if(max_degree < S)
            max_degree = S;
    }

    // Get the set of neighbors for each node
    neighbors.resize(numberOfNodes);
    for(size_t i = 0; i < numberOfNodes; i++){
        for(size_t idx = 0; idx < containing_factors[i].size(); idx++){
            // All factor containing i
            size_t c = containing_factors[i][idx];
            for(size_t jdx = 0; jdx < factors[c].size(); jdx++){
                size_t j = factors[c][jdx];
                neighbors[i].insert(j);
            }
        }
    }


    absMax = 0;
    for(size_t c = 0; c < factors.size(); c++){
        for(size_t idx = 0; idx < potentials[c].size(); idx++){
            double pmax = std::abs(potentials[c][idx]);
            if(absMax < pmax)
                absMax = pmax;
        }
    }


    TensorIndicesOfAllRanks.clear();
    TensorIndicesOfAllRanks.resize(max_degree);
    for(size_t d = 0; d < max_degree; d++){
        getTensorIndices(TensorIndicesOfAllRanks[d], numberOfLabels[0], d+1);
    }

}


#endif // HMRF_H
