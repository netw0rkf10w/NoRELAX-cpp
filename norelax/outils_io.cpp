#include <iostream>
#include <fstream>
#include <iterator>
#include <chrono>
#include <sstream>
#include <assert.h>
//#include <unsupported/Eigen/CXX11/Tensor>
#include "outils_io.hpp"


using namespace std;
//using namespace Eigen;


//int ReadNumbers_CV( const string & s, cv::Mat& v ) {
//    istringstream is( s );
//    float n;
//    while( is >> n ) {
//        v.push_back( n );
//    }
//    return max(v.rows, v.cols);
//}



//void ReadMatrixFile_CV(const char* filename_X, cv::Mat &m, int& rows, int& cols)
///// Read the matrix file, set rows and cols, return a CV_32F matrix
//{
//    ifstream file_X;
//    string line;

//    file_X.open(filename_X);
//    assert (file_X.is_open());

//    int i=0;
//    cv::Mat firstRow;
//    getline(file_X, line);
//    cols = ReadNumbers_CV( line, firstRow );
//    m.release();
//    m.push_back(firstRow.t());

//    for ( i=1; i < INT_MAX; i++)
//    {
//        if ( !getline(file_X, line) ) break;
//        cv::Mat aRow;
//        if( ReadNumbers_CV(line, aRow) == cols )
//            m.push_back(aRow.t());
//        else
//        {
//            cerr<<"The number of elements of row "<<i<<" does not match the number of columns!!!"<<endl;
//            assert(false);
//        }
//    }
//    file_X.close();
//    rows = i;
//    assert(rows < INT_MAX);
//    assert(rows == m.rows && cols == m.cols);
//}




//void ReadProposals_CV(std::vector<cv::Mat> &proposals, const std::string& filename)
///// Read the (unary) data terms and the disparities of the proposals
///// Proposal file:
///// row 0: height width 0 0 ....
///// From row 1: each row corresponds to a proposal (either disparity or unary cost, depending on the input file), COLUMN MAJOR
//{
//    proposals.clear();
//    cv::Mat m;
//    int rows, cols;
//    ReadMatrixFile_CV(filename.c_str(), m, rows, cols);

//    int height = m.at<float>(0,0);
//    int width = m.at<float>(0,1);
//    int numProposals = rows - 1;
//    proposals.reserve(numProposals);

//    for(int i = 0; i < numProposals; i++){
//        // IMPORTANT: OpenCV matrix is ROW-MAJOR
//        cv::Mat M = m.row(i+1).reshape(0, width);
//        M = M.t();
//        assert(height == M.rows && width == M.cols);
//        proposals.push_back(M);
//    }
//}
