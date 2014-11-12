/*

EigenFaceTest.cpp
Description:   Interface to run Eigenface tests

Author:        Chris Leighton
Date:          June 27th 2011

*/

//#ifdef EIGENFACE_TEST

#include <algorithm>
#include <cctype>
#include <string>
#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <highgui.h>
#include "FaceDetector.h"
#include "Utilities.h"
#include "PreProcess.h"
#include "Training.h"
#include "TrainingFile.h"
#include "Recognize.h"
#include "KMeans.h"
#include "UPGMA.h"

void PrintUsage();

int main( int argc, char** argv )
{
    std::string command = "";

    if ( argc <= 4 )
    {
        PrintUsage();
        return 1;
    }

    std::string trainFile(argv[1]);
    std::string testFile(argv[2]);
    std::string databaseName(argv[3]);
    std::string strResultsFile(argv[4]);
    std::string resultsDir = "./";

    std::ofstream resultsFile(strResultsFile.c_str());
    if ( !resultsFile.is_open() )
    {
        cout << "Could not open results file" << endl;
        return 1;
    }

    resultsFile << "Results for Test run at " << getDateTime() << endl;
    resultsFile << "Training File: " << trainFile << endl;
    resultsFile << "Test File    : " << testFile << endl;
    resultsFile << "---------------------------------------" << endl;

    int train_ms;
    int test_ms;
    std::stringstream details;
    double t = 0.0;

    try
    {
        /*/////////////////////////////  Training //////////////////////////////////////////
        cout << "Starting the training procress" << endl;

        t = (double)cvGetTickCount();
        Train( trainFile.c_str(), databaseName.c_str(), resultsDir );
        t = (double)cvGetTickCount() - t;
        train_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );

        cout << "Database created: " << databaseName << endl;
        cout << endl;

        ////////////////////////////////////////////////////////////////////////////////////*/

        /*///////////////////////////// Eigenface recognition /////////////////////////////////////////
        cout << "Starting Recognition Test" << endl;

        int totalTested = 0;
        int totalFound = 0;
        int totalNotFound = 0;

        std::ifstream in(testFile.c_str());
        if ( !in.is_open() )
            throw std::string("Could not open test file");

        t = (double)cvGetTickCount();
        char linebuffer[512];
        Database* db = new Database();
        db->Read(databaseName);

        while (in.getline(linebuffer,512))
        {
            std::string line(linebuffer);
            size_t pos1;
            size_t pos2;

            int trueid = 0;
            std::string personName = "";
            std::string probeFace = "";

            // true person ID
            pos1 = line.find_first_of(' ');
            trueid = atoi(line.substr(0, pos1).c_str());

            // person name
            pos2 = pos1;
            pos1 = line.find(' ', pos2+1);
            personName = line.substr(pos2+1, pos1-pos2);

            probeFace = line.substr(pos1+1, line.length()-pos1+1);

            cout << "Attempting to recognize true id " << trueid << " name: " << personName << " image: "
                 << probeFace << endl;

            double distance = 0.0;

            int idFound = 0;

            std::string result = Recognize(probeFace.c_str(), databaseName.c_str(), distance, resultsDir, idFound, db, false );
            totalTested++;

            cout << "Id found " << idFound << endl;

            if ( trueid != idFound )
            {
                totalNotFound++;
                cout << "Could not find person" << endl;
            }
            else
            {
                totalFound++;
                cout << "Found: " << result << endl;
            }
            cout << "Distance: " << distance << endl << endl;

        }

        t = (double)cvGetTickCount() - t;
        test_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );

        resultsFile << "Total Tested   : " << totalTested << endl;
        resultsFile << "Total Found    : " << totalFound  << endl;
        resultsFile << "Total Not Found: " << totalNotFound << endl;
        resultsFile << "Train time(ms) : " << train_ms << endl;
        resultsFile << "Test time (ms) : " << test_ms << endl;

        // TODO - reflect false positive and false negative results in results
        double percentFound = 100.0 * ((double)totalFound/(double)totalTested);
        double percentNFound = 100.0 * ((double)totalNotFound/(double)totalTested);
        resultsFile << "Percent Found     : " << percentFound << endl;
        resultsFile << "Percent Not Found : " << percentNFound << endl;
        in.close();
        delete db;
        /////////////////////////////////////////////////////////////////////////////////////////*/

        /*////////////// do KMeans on original images ////////////////////////
        t = (double)cvGetTickCount();
        cout << "Starting KMeans on original images" << endl;
        resultsFile << "Starting KMeans on original images" << endl;
        KMeans1(databaseName, resultsFile);
        t = (double)cvGetTickCount() - t;
        test_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
        resultsFile << "Performed KMeans on original images in " << test_ms << " ms." << endl;
        ///////////////////////////////////////////////////////////////////// */

        UPGMA upgma(databaseName.c_str());
        std::string steps;
        /*/////////////// do UPGMA on original images with Euclidean Disance Coefficient ////////////////////////
        t = (double)cvGetTickCount();
        cout << "Starting UPGMA on original images with Euclidean Distance Coefficient" << endl;
        resultsFile << "Starting UPGMA on original images with Euclidean Distance Coefficient" << endl;

        upgma.LoadImages();
        upgma.DoCluster(EuclideanDistanceCoefficient);
        upgma.GetStrClusterSteps(steps, false);
        //std::cout << steps;

        std::string clustersA;
        upgma.GetClustersAtClusterCount(upgma.GetnPeople(), clustersA);
        resultsFile << clustersA;

        t = (double)cvGetTickCount() - t;
        test_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
        resultsFile << "Performed UPGMA on original images with Euclidean Distance Coefficient in " << test_ms << " ms." << endl;
        ///////////////////////////////////////////////////////////////////// */


        /*/////////////// do UPGMA on Reduced images with Euclidean Disance Coefficient ////////////////////////
        t = (double)cvGetTickCount();
        cout << "Starting UPGMA on Reduced images with Euclidean Distance Coefficient" << endl;
        resultsFile << "Starting UPGMA on Reduced images with Euclidean Distance Coefficient" << endl;

        upgma.LoadReducedImages();
        upgma.DoCluster(EuclideanDistanceCoefficient);
        upgma.GetStrClusterSteps(steps, false);
        //std::cout << steps;

        std::string clustersB;
        upgma.GetClustersAtClusterCount(upgma.GetnPeople(), clustersB);
        resultsFile << clustersB;

        t = (double)cvGetTickCount() - t;
        test_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
        resultsFile << "Performed UPGMA on Reduced images with Euclidean Distance Coefficient in " << test_ms << " ms." << endl;
        ///////////////////////////////////////////////////////////////////// */

        /*/////////////// do UPGMA on original images with Coefficient of shape diff ////////////////////////
        t = (double)cvGetTickCount();
        cout << "Starting UPGMA on original images with Coefficient of shape diff" << endl;
        resultsFile << "Starting UPGMA on original images with Coefficient of shape diff" << endl;

        upgma.LoadImages();
        upgma.DoCluster(CoefficientOfShapeDiff);
        upgma.GetStrClusterSteps(steps, false);
        //std::cout << steps;

        std::string clustersC;
        upgma.GetClustersAtClusterCount(upgma.GetnPeople(), clustersC);
        resultsFile << clustersC;

        t = (double)cvGetTickCount() - t;
        test_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
        resultsFile << "Performed UPGMA on original images with Coefficient of shape diff in " << test_ms << " ms." << endl;
        ///////////////////////////////////////////////////////////////////// */

        //////////////// do UPGMA on original images with Cosine Coefficient ////////////////////////
        t = (double)cvGetTickCount();
        cout << "Starting UPGMA on original images with Cosine Coefficient" << endl;
        resultsFile << "Starting UPGMA on original images with Cosine Coefficient" << endl;

        upgma.LoadImages();
        upgma.DoCluster(CosineCoefficient);
        upgma.GetStrClusterSteps(steps, false);
        //std::cout << steps;

        std::string clustersD;
        upgma.GetClustersAtClusterCount(upgma.GetnPeople(), clustersD);
        resultsFile << clustersD;

        t = (double)cvGetTickCount() - t;
        test_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
        resultsFile << "Performed UPGMA on original images with Cosine Coefficient " << test_ms << " ms." << endl;
        ///////////////////////////////////////////////////////////////////// */

        /*/////////////// do UPGMA on original images with Correlation Coefficient ////////////////////////
        t = (double)cvGetTickCount();
        cout << "Starting UPGMA on original images with Correlation Coefficient" << endl;
        resultsFile << "Starting UPGMA on original images with Correlation Coefficient" << endl;

        upgma.LoadImages();
        upgma.DoCluster(CorrelationCoefficient);
        //upgma.GetStrClusterSteps(steps, false);
        //std::cout << steps;

        std::string clustersE;
        upgma.GetClustersAtClusterCount(upgma.GetnPeople(), clustersE);
        resultsFile << clustersE;

        t = (double)cvGetTickCount() - t;
        test_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
        resultsFile << "Performed UPGMA on original images with Corellation Coefficient " << test_ms << " ms." << endl;
        ///////////////////////////////////////////////////////////////////// */


        /*/////////////// do UPGMA on original images with Canberra Metric Coefficient ////////////////////////
        t = (double)cvGetTickCount();
        cout << "Starting UPGMA on original images with Canberra Metric Coefficient" << endl;
        resultsFile << "Starting UPGMA on original images with Canberra Metric Coefficient" << endl;

        upgma.LoadImages();
        upgma.DoCluster(CanberraMetricCoefficient);
        upgma.GetStrClusterSteps(steps, false);
        //std::cout << steps;

        std::string clustersF;
        upgma.GetClustersAtClusterCount(upgma.GetnPeople(), clustersF);
        resultsFile << clustersF;

        t = (double)cvGetTickCount() - t;
        test_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
        resultsFile << "Performed UPGMA on original images with Canberra Metric Coefficient " << test_ms << " ms." << endl;
        ///////////////////////////////////////////////////////////////////// */



        /*/////////////// do UPGMA on original images with Bray-Curtis Coefficient ////////////////////////
        t = (double)cvGetTickCount();
        cout << "Starting UPGMA on original images with Bray-Curtis Coefficient" << endl;
        resultsFile << "Starting UPGMA on original images with Bray-Curtis Coefficient" << endl;

        upgma.LoadImages();
        upgma.DoCluster(BrayCurtisCoefficient);
        upgma.GetStrClusterSteps(steps, false);
        //std::cout << steps;

        std::string clustersG;
        upgma.GetClustersAtClusterCount(upgma.GetnPeople(), clustersG);
        resultsFile << clustersG;

        t = (double)cvGetTickCount() - t;
        test_ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
        resultsFile << "Performed UPGMA on original images with Bray-Curtis Coefficient " << test_ms << " ms." << endl;
        ///////////////////////////////////////////////////////////////////// */

        resultsFile.close();
    }
    catch ( std::string err )
    {
        cout << "Error: " << err << endl;
        return 1;
    }

    return 0;
}




void PrintUsage()
{
    cout << "EigenFace train.dat test.dat [database to write to] [results file]"  << std::endl;
}

//#endif // EIGENFACE_TEST
