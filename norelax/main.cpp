#include <iostream>
#include <chrono>
#include <string>
#include <set>
#include <iomanip>
#include <fstream>
#include "makeunique.hpp"
#include "cxxopts.hpp"

#include "OpenGM.hpp"
//#include "unittest.h"
#include "HMRF.hpp"
#include "PairwiseMRF.hpp"
#include "DenseMRF.hpp"

using namespace std;
using namespace Eigen;

void inference_caller(const std::string &method, const std::string &model, const std::string &output, bool verbose,
              double rho, double step, int iter1, int iter2, int maxIt,
              int protocolFrequence, double precision, double timeout, int model_type = 0, bool lineSearch = true, int numInit = 1);

template <typename GraphicalModelType>
std::unique_ptr<HMRF> CreateModel(int model_type, const GraphicalModelType &gm);
std::unique_ptr<HMRF> CreateModel(int model_type, const string &model);



void test_stationary_point()
// A point x* is stationary if and only if min_x <G(x*), x> = <G(x*), x*> where G(x^*) is the gradient of the MRF energy at x*.
{
    string datapath("/home/khue/Research/MRFs/models/");
    GM gm;
    //opengm::hdf5::load(gm, datapath + "cell-tracking/ogm_model.h5","gm");
    opengm::hdf5::load(gm, datapath + "inclusion/modelH-1-0.8-0.2.h5","gm");
    //opengm::hdf5::load(gm, datapath + "mrf-stereo/tsu-gm.h5","gm");
    //opengm::hdf5::load(gm, datapath + "protein-prediction/didNotconverge1.h5","gm");
    //opengm::hdf5::load(gm, "/home/khue/Research/Stereo/data/tsukuba/tsukuba_small.h5","gm");


    /// Import to our model
    cout<<"Import to our model..."<<endl;
    HMRF mrf;
    mrf.importModel(gm);


    /// Check if the solution returend by ICM is stationary
   /* // Solve with ICM (= BCD)
    cout<<"Solving with ICM..."<<endl;
    typedef opengm::ICM<GM, opengm::Minimizer> ICM;
    ICM solver(gm);
    ICM::VerboseVisitorType visitor;
    solver.infer(visitor);
    // Get the labeling and the energy
    vector<LabelType> label_initial;
    solver.arg(label_initial);

    size_t V = gm.numberOfVariables();
    size_t L = gm.numberOfLabels(0);
    MatrixXd X0 = MatrixXd::Zero(L,V);
    for(size_t i = 0; i < V; i++){
        size_t l = label_initial[i];
        X0(l, i) = 1.0;
    }
    // Assign the current solution to the solution returned by ICM
    mrf.setX(X0);*/


    /// Check if the solution returend by ADMM is stationary
    myclock::time_point begin, end;
    double timeSec;

    HMRF::Parameters param;
    param.rho_min = 0.001;
    param.rho_max = 50.0;
    param.step = 1.2;
    param.iter1 = 300;
    param.iter2 = 200;
    param.MAX_ITER = 100000;
    param.verbose = true;
    param.protocolFrequence = 50;
    param.precision = 1e-5;
    mrf.setParameters(param);

    begin = myclock::now();

    //
    //mrf.FW();
    //mrf.PGD();
    //
    //
    mrf.ADMM();
    //mrf.FW();
    mrf.BCD();
    //

    end = myclock::now();
    timeSec = measure_time(begin, end);
    cout << "Time: " << timeSec << "s" << endl;


    /// Check if the current x is stationary
    // Find s = argmin_x <G(x*), x>
    MatrixXd G;
    mrf.gradient(G);
    MatrixXd S;
    S.setZero(G.rows(), G.cols());
    for(size_t i = 0; i < (size_t)G.cols(); i++){
        size_t l_min;
        G.col(i).minCoeff(&l_min);
        S(l_min, i) = 1.0;
    }

    // Check if <G(x*), s> = <G(x*), x*>
    double A = (G.cwiseProduct(S)).sum();
    double B = (G.cwiseProduct(mrf.getX())).sum();

    cout<<"<G(x*), s> = "<<A<<endl<<"<G(x*), x*> = "<<B<<endl;
    cout<<"Gap = "<<B-A<<endl;
}






void test_highorder()
{
    string datapath("/home/khue/Research/MRFs/models/");
    GM gm;
    //opengm::hdf5::load(gm, datapath + "cell-tracking/ogm_model.h5","gm");
    opengm::hdf5::load(gm, datapath + "inclusion/modelH-6-0.8-0.2.h5","gm");
    //opengm::hdf5::load(gm, datapath + "mrf-stereo/tsu-gm.h5","gm");
    //opengm::hdf5::load(gm, datapath + "protein-prediction/didNotconverge1.h5","gm");
    //opengm::hdf5::load(gm, "/home/khue/Research/Stereo/data/tsukuba/tsukuba_small.h5","gm");

    /// Import to our model
    HMRF mrf;
    mrf.importModel(gm);

    myclock::time_point begin, end;
    double timeSec;

    double energy;

    HMRF::Parameters param;
    param.rho_min = 0.00001;
    param.rho_max = 50.0;
    param.step = 1.2;
    param.iter1 = 500;
    param.iter2 = 300;
    param.MAX_ITER = 200000;
    param.verbose = true;
    param.protocolFrequence = 50;

    mrf.setParameters(param);


    //VD energies, bounds, residuals, times;
    //VS states, iteration;

    begin = myclock::now();

    mrf.inference();

    //mrf.cgd();

    //mrf.ADMM();

    end = myclock::now();
    timeSec = measure_time(begin, end);
    cout << "Time: " << timeSec << "s" << endl;

//    mrf.cgd();

//    end = myclock::now();
//    timeSec = measure_time(begin, end);
//    cout << "Time: " << timeSec << "s" << endl;




    VS labeling(mrf.getNumberOfNodes(), 0);
    MatrixXd X;

    X = mrf.getX();
    for(size_t i = 0; i < mrf.getNumberOfNodes(); i++){
        size_t label_max;
        X.col(i).maxCoeff( &label_max );
        labeling[i] = label_max;
    }
    energy = mrf.energy();
    cout<<"Continuous energy = "<<setprecision(10)<<energy<<endl;
    cout<<"OpenGM energy = "<<setprecision(10)<<gm.evaluate(labeling)<<endl;


//    // Save the optimal labeling to file
//    string path("/home/khue/Research/Stereo/data/tsukuba/");
//    std::ofstream output_file(path + "tsukuba_small_norelax.txt");
//    if(output_file)
//    {
//        ostream_iterator<int> output_iterator(output_file, "\n");
//        copy(labeling.begin(), labeling.end(), output_iterator);
//    }else{
//        cout<<"Could not open file"<<endl;
//    }
}




void test_pairwise()
{
    string datapath("/home/khue/Research/MRFs/models/");
    GM gm;
    //opengm::hdf5::load(gm, datapath + "cell-tracking/ogm_model.h5","gm");
    //opengm::hdf5::load(gm, datapath + "inclusion/modelH-1-0.8-0.2.h5","gm");
    //opengm::hdf5::load(gm, datapath + "mrf-stereo/tsu-gm.h5","gm");
    //opengm::hdf5::load(gm, datapath + "protein-prediction/didNotconverge1.h5","gm");
    //opengm::hdf5::load(gm, "/home/khue/Research/Stereo/data/tsukuba/tsukuba_small.h5","gm");
    opengm::hdf5::load(gm, datapath + "inpainting-n8/triplepoint4-plain-ring-inverse.h5","gm");

    myclock::time_point begin, end;
    double timeSec;

    double energy;


    // Solve with ICM (= BCD)
//    cout<<"Solving with ICM..."<<endl;
//    typedef opengm::ICM<GM, opengm::Minimizer> ICM;
//    ICM solver(gm);
//    ICM::VerboseVisitorType visitor;
//    begin = myclock::now();
//    solver.infer(visitor);
//    // Get the labeling and the energy
//    vector<LabelType> label_initial;
//    solver.arg(label_initial);
//    end = myclock::now();
//    timeSec = measure_time(begin, end);
//    cout << "Time: " << timeSec << "s" << endl;


    PairwiseSharedMRF mrf;
    //PairwiseMRF mrf;
    //HMRF mrf;

    HMRF::Parameters param;
    param.rho_min = 0.00001;
    param.rho_max = 50.0;
    param.step = 1.2;
    param.iter1 = 500;
    param.iter2 = 300;
    param.MAX_ITER = 200000;
    param.verbose = true;
    param.protocolFrequence = 50;
    param.precision = 1e-15;

    mrf.setParameters(param);

    /// Import to our model
    mrf.importModel_Pairwise_Metric(gm);
    //mrf.importModel_Pairwise(gm);
    //mrf.importModel(gm);





    //VD energies, bounds, residuals, times;
    //VS states, iteration;

    begin = myclock::now();

    mrf.inference();

    //mrf.cgd();

    //mrf.ADMM();

    end = myclock::now();
    timeSec = measure_time(begin, end);
    cout << "Time: " << timeSec << "s" << endl;


    VS labeling(mrf.getNumberOfNodes(), 0);
    MatrixXd X;

    X = mrf.getX();
    for(size_t i = 0; i < mrf.getNumberOfNodes(); i++){
        size_t label_max;
        X.col(i).maxCoeff( &label_max );
        labeling[i] = label_max;
    }
    energy = mrf.energy();
    cout<<"Discrete energy = "<<setprecision(10)<<energy<<endl;
    cout<<"OpenGM energy = "<<setprecision(10)<<gm.evaluate(labeling)<<endl;
}






//void test_stereo()
//{
//    string basename = "/home/khue/Research/Stereo/data/tsukuba";
//    string unariesFile = basename + "/unaries.txt";
//    string proposalsFile = basename + "/proposals.txt";
//    string horizontalWeightsFile = basename + "/weights_horizontal.txt";
//    string verticalWeightsFile = basename + "/weights_vertical.txt";

//    float sigma = 0.02;
//    float smoothScale = 100;

//    HMRF mrf;
//    mrf.importStereoModel(sigma, smoothScale, proposalsFile, unariesFile, horizontalWeightsFile, verticalWeightsFile);

//    myclock::time_point begin, end;
//    double timeSec;

//    double energy;

//    HMRF::Parameters param;
//    param.rho_min = 10.0;
//    param.rho_max = 50.0;
//    param.step = 1.2;
//    param.iter1 = 500;
//    param.iter2 = 300;
//    param.MAX_ITER = 100;
//    param.verbose = true;
//    param.protocolFrequence = 10;

//    mrf.setParameters(param);


//    //VD energies, bounds, residuals, times;
//    //VS states, iteration;

//    begin = myclock::now();

//    //mrf.inference();

//    //mrf.cgd();

//    mrf.ADMM();

//    end = myclock::now();
//    timeSec = measure_time(begin, end);
//    cout << "Time: " << timeSec << "s" << endl;

////    mrf.cgd();

////    end = myclock::now();
////    timeSec = measure_time(begin, end);
////    cout << "Time: " << timeSec << "s" << endl;

//    mrf.BCD();

//    end = myclock::now();
//    timeSec = measure_time(begin, end);
//    cout << "Time: " << timeSec << "s" << endl;


//    VS labeling(mrf.getNumberOfNodes(), 0);
//    MatrixXd X;

//    X = mrf.getX();
//    for(size_t i = 0; i < mrf.getNumberOfNodes(); i++){
//        size_t label_max;
//        X.col(i).maxCoeff( &label_max );
//        labeling[i] = label_max;
//    }
//    energy = mrf.energy();
//    cout<<"Discrete energy = "<<setprecision(10)<<energy<<endl;


//    // Save the optimal labeling to file
//    string path("/home/khue/Research/Stereo/data/tsukuba/");
//    std::ofstream output_file(path + "tsukuba_small_norelax.txt");
//    if(output_file)
//    {
//        ostream_iterator<int> output_iterator(output_file, "\n");
//        copy(labeling.begin(), labeling.end(), output_iterator);
//    }else{
//        cout<<"Could not open file"<<endl;
//    }
//}




int main(int argc, char** argv)
{
    //checkLabelIndices();
    //checkTensorIndices();
    //checkDataConversion();
    //test_stereo();
    //test_stationary_point();
    //test_pairwise();

    //test_highorder();
    //checkProposalReading();

    // Common parameters
    int maxIt = 200000;
    bool verbose = false;
    int protocolFrequence = 50;
    double precision = 1e-4;
    double timeout = 3600;

    // ADMM parameters
    double rho = -1;
    double step = 1.2;
    int iter1 = 300;
    int iter2 = 200;

    // PDG/FW/CQP parameters
    bool lineSearch = false;

    // Number of initial solutions (valid for BCD/PGD/FW/CQP)
    int numInit = 1;
    int numThreads = -1;
    int model_type = MRF_TYPE_HIGHORDER;

    std::string method("admm");
    std::string model("/home/khue/Research/MRFs/models/inclusion/modelH-10-0.8-0.2.h5");
    std::string output("modelH-10-0.8-0.2.h5");

    try
      {
        cxxopts::Options options(argv[0], "***********************************\nNoRELAX - A generic solver for MAP inference and MRF optimization");

        options.add_options()
                ("v,verbose", "Verbose mode", cxxopts::value<bool>(verbose))
                ("opt", "Optimization method: admm, bcd (block coordinate descent), pgd (projected gradient descent), fw (frank-wolfe), cqp (convex QP relaxation)", cxxopts::value<std::string>(method), "[admm, bcd, pgd, fw]")
                ("numInit", "Number of initial solutions", cxxopts::value<int>(numInit), "<int>")
                ("m,model", "Path to input graphical model file (*.h5)", cxxopts::value<std::string>(model), "<path.h5>")
                ("o,output", "Path to output file (*.h5)", cxxopts::value<std::string>(output), "<path.h5>")
                ("maxIt", "Max number of iterations", cxxopts::value<int>(maxIt), "<int>")               
                ("protocol", "Frequency of calculating the energy", cxxopts::value<int>(protocolFrequence), "<int>")
                ("p,precision", "Desired precision (residual for ADMM, gradient norm for PGD, FW gap for Frank-Wolfe)", cxxopts::value<double>(precision), "<double>")
                ("timeout", "Timeout in seconds", cxxopts::value<double>(timeout), "<double>")
                ("type", "MRF type. 0: higher-order (default), 1: pairwise, 2: shared pairwise, 3: dense mrf.", cxxopts::value<int>(model_type), "<int>")
                ("h,help", "Print help")
                ("r,rho", "Initial value of the penalty parameter rho", cxxopts::value<double>(rho), "<double>")
                ("s,step", "Step size for penalty update", cxxopts::value<double>(step), "<double>")
                ("iter1", "Time (no. of iterations) for stability after penalty update (c.f. paper)", cxxopts::value<int>(iter1), "<int>")
                ("iter2", "Frequency of checking for descent (c.f. paper)", cxxopts::value<int>(iter2), "<int>")
                ("linesearch", "Use line search for gradient-based methods", cxxopts::value<bool>(lineSearch))
                ("threads", "Number of parallel threads", cxxopts::value<int>(numThreads), "<int>")
            #ifdef CXXOPTS_USE_UNICODE
                  ("unicode", u8"A help option with non-ascii: à. Here the size of the"
                    " string should be correct")
                #endif
                ;

        //options.parse_positional({"input", "output", "positional"});
        options.parse(argc, argv);

        if (options.count("help"))
        {
          std::cout << options.help({"", "Group"}) << std::endl;
          exit(0);
        }

        if(!options.count("model"))
        {
            std::cout << "Option 'model' is required!" << std::endl;
            exit(1);
        }

        //cout<<"verbose = "<<verbose<<", rho = "<<rho<<", step = "<<step<<", maxIt = "<<maxIt<<endl;
        //cout<<"iter1 = "<<iter1<<", iter2 = "<<iter2<<", protocol = "<<protocolFrequence<<", precision = "<<precision<<endl;
        //cout<<"input = "<<input<<endl<<"output = "<<output<<endl;

    }catch (const cxxopts::OptionException& e)
    {
      std::cout << "error parsing options: " << e.what() << std::endl;
      exit(1);
    }

    /// END Passing arguments
    /// **************************

    if(numThreads > 0){
        omp_set_num_threads(numThreads);
    }
    inference_caller(method, model, output, verbose, rho, step, iter1, iter2, maxIt, protocolFrequence, precision, timeout, model_type, lineSearch, numInit);

    return 0;
}





void inference_caller(const std::string &method, const std::string &input, const std::string &output, bool verbose,
              double rho, double step, int iter1, int iter2, int maxIt,
              int protocolFrequence, double precision, double timeout, int model_type, bool lineSearch, int numInit)
{
    /// Import to HMRF model
    if(verbose)
        cout<<"Importing from "<<input<<endl;
    //GM gm;
    //opengm::hdf5::load(gm, model,"gm");
    //auto mrf = CreateModel(model_type, gm);

    auto mrf = CreateModel(model_type, input);
    if(verbose)
        cout<<"Model imported."<<endl;

    /// Passing the parameters

    HMRF::Parameters param;
    param.rho_min = rho;
    //param.rho_max = 100.0;
    param.step = step;
    param.iter1 = iter1;
    param.iter2 = iter2;
    param.MAX_ITER = maxIt;
    param.verbose = verbose;
    param.protocolFrequence = protocolFrequence;
    param.precision = precision;
    param.timeout = timeout;
    param.lineSearch = lineSearch;
    param.numInit = numInit;

    mrf->setParameters(param);
    mrf->setOutputLocation(output);



    myclock::time_point begin, end;
    double timeSec;

    begin = myclock::now();
    mrf->inference(method);
    end = myclock::now();

    timeSec = measure_time(begin, end);
    cout << "Inference time: " << timeSec << "s" << endl;
    cout<<"Energy = "<<setprecision(10)<<mrf->energy()<<endl;
    //cout<<"Discrete energy = "<<setprecision(10)<<gm.evaluate(mrf.getStates())<<endl;

    //GM gm;
    //opengm::hdf5::load(gm, input, "gm");
    //cout<<"OpenGM energy = "<<setprecision(10)<<gm.evaluate(mrf->getStates())<<endl;

}



template <typename GraphicalModelType>
std::unique_ptr<HMRF> CreateModel(int model_type, const GraphicalModelType &gm)
{
    switch(model_type)
    {
        case 1:
        {
            auto model = std::make_unique<PairwiseMRF>();
            model->importModel_Pairwise(gm);
            return model; // or std::move(model)
        }
        case 2:
        {
            auto model = std::make_unique<PairwiseSharedMRF>();
            model->importModel_Pairwise_Metric(gm);
            return model; // or std::move(model)
        }
        case 3:
        {
            auto model = std::make_unique<DenseMRF>();
            model->importModel_Dense(gm);
            return model; // or std::move(model)
        }
        default:
        {
            auto model = std::make_unique<HMRF>();
            model->importModel(gm);
            return model;
        }
    }
}


std::unique_ptr<HMRF> CreateModel(int model_type, const string &modelFile)
{
    GM gm;
    opengm::hdf5::load(gm, modelFile, "gm");

    switch(model_type)
    {
        case 1:
        {
            auto model = std::make_unique<PairwiseMRF>();
            model->importModel_Pairwise(gm);
            return model; // or std::move(model)
        }
        case 2:
        {
            auto model = std::make_unique<PairwiseSharedMRF>();
            model->importModel_Pairwise_Metric(gm);
            return model; // or std::move(model)
        }
        case 3:
        {
            auto model = std::make_unique<DenseMRF>();
            model->importModel_Dense(gm);
            return model; // or std::move(model)
        }
        default:
        {
            auto model = std::make_unique<HMRF>();
            model->importModel(gm);
            return model;
        }
    }
}
