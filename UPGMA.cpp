#include "UPGMA.h"
#include "ResemblanceCoefficient.h"
#include <algorithm>


UPGMA::UPGMA( const char* databaseName ) : m_pDataMatrix(NULL), m_pResemblanceMatrix(NULL), m_pOriginalResemblanceMatrix(NULL),
                                           m_bIsDisimilarityCoeffcient(true), m_pCopheneticMatrix(NULL), m_Threshold(0.0)
{
    try
    {
        m_pDatabase = NULL;
        m_pDatabase = new Database();
        std::string dbName(databaseName);

        if ( !m_pDatabase->Read(databaseName) )
            throw std::string("UPGMA Constructor could not create database");

        m_nPeople = m_pDatabase->GetnPeople();
        m_nObjects = m_nAttributes = 0;

    }
    catch (...)
    {
        throw;
    }

}


bool UPGMA::LoadImages()
{
    bool bRet = true;

    Clear();

    Database::ImageVec& imageVec = m_pDatabase->GetImageVec();
    m_nObjects = m_pDatabase->GetnImages();

    // get dimensions of original images
    CvSize imageCvSize;
    imageCvSize.width = imageVec[0].m_Image->width;
    imageCvSize.height = imageVec[0].m_Image->height;
    m_nAttributes = imageCvSize.width * imageCvSize.height;

    // store original images in CvMat - each row contains a width*height image
    m_pDataMatrix = cvCreateMat(m_nObjects, m_nAttributes, CV_32FC1);
    for ( int i = 0; i < m_nObjects; i++ )
    {
        IplImage* img = imageVec[i].m_Image;

        // iterate each image, store intensity in originalImages
        for ( int row = 0; row < img->height; row++ )
        {
            for ( int col = 0; col < img->width; col++ )
            {
                CvScalar s;
                s = cvGet2D(img, row, col);
                int colInOriginals = row*img->height+col;
                int rowInOriginals = i*m_nAttributes;
                int posInOriginals = rowInOriginals + colInOriginals;
                m_pDataMatrix->data.fl[posInOriginals] = (double)s.val[0];
            }
        }
    }

    return bRet;
}


bool UPGMA::LoadReducedImages()
{
    bool bRet = true;

    Clear();

    m_nObjects = m_pDatabase->GetnImages();
    m_nAttributes = m_pDatabase->GetnEigenVals();

    m_pDataMatrix = cvCreateMat(m_nObjects, m_nAttributes, CV_32FC1);
    for ( int i = 0; i < m_nObjects*m_nAttributes; i++ )
        m_pDataMatrix->data.fl[i] = projectedFaceMatrix->data.fl[i];

    return bRet;
}


bool UPGMA::DoCluster( ResemblanceCoefficientType t )
{
    bool bRet = true;

    m_bIsDisimilarityCoeffcient = IsDisimilarType(t);
    m_pResemblanceMatrix = cvCreateMat( m_nObjects, m_nObjects, CV_32FC1 );
    m_pOriginalResemblanceMatrix = cvCreateMat( m_nObjects, m_nObjects, CV_32FC1 );

    m_pCopheneticMatrix = cvCreateMat( m_nObjects, m_nObjects, CV_32FC1 );
    for ( int i = 0; i < m_nObjects*m_nObjects; i++ )
        m_pCopheneticMatrix->data.fl[i] = std::numeric_limits<double>::max();

    try
    {
        // Find initial resemblance matrix
        CalcResemblanceMatrix(t, m_nObjects);
        for ( int i = 0; i < m_nObjects*m_nObjects; i++ )
            m_pOriginalResemblanceMatrix->data.fl[i] = m_pResemblanceMatrix->data.fl[i];

        m_Threshold = GetAverageResemblance(this, m_nObjects);
        int nCurObjects = m_nObjects;

        // at step 0, each object is a cluster
        Cluster_step step0;
        for ( int i = 0; i < m_nObjects; i++ )
        {
            Cluster cluster;
            cluster.objects.push_back(i);
            cluster.distance = 0.0;
            cluster.bIsNew = true;
            step0.clusters.push_back(cluster);
            m_ResemblanceLables.push_back(cluster);
        }
        m_Steps.push_back(step0);

        for ( int step = 1; step < m_nObjects; step++ )
        {
            double val;
            int object1;
            int object2;
            Cluster_step stepN;
            Cluster      newcluster;
            newcluster.bIsNew = true;

            GetResemblanceValue(this, object1, object2,val);
            // merging two clusters together
            // need to update the following
            // 1. Add this cluster to the new cluster step and store in m_Steps
            // 2. Update the Resemblancelables to reflect that we 'ate' away two clusters to form 1
            // 3. Update the ResemblanceMatrix to reflect this step as well

            // 1. update m_Steps
            Cluster c1 = m_ResemblanceLables[object1];
            Cluster c2 = m_ResemblanceLables[object2];
            for ( size_t i = 0; i < c1.objects.size(); i++ )
                newcluster.objects.push_back(c1.objects[i]);
            for ( size_t i = 0; i < c2.objects.size(); i++ )
                newcluster.objects.push_back(c2.objects[i]);

            newcluster.distance = val;

            std::vector<Cluster> clusters = m_Steps[step-1].clusters;
            size_t size = clusters.size();
            for ( size_t i = 0; i < size; i++ )
            {
                if ( clusters[i] != c1 && clusters[i] != c2 )
                {
                    Cluster tempcluster;
                    tempcluster.objects = clusters[i].objects;
                    tempcluster.distance = clusters[i].distance;
                    tempcluster.bIsNew = false;
                    stepN.clusters.push_back(tempcluster);
                }
            }
            stepN.clusters.push_back(newcluster);
            m_Steps.push_back(stepN);

            // 2. Update Resemblancelables by removing the clusters at
            // object1 and object2 and adding the new cluster at object1's pos
            std::vector<Cluster> newResemblanceLables;

            // erase the highest value first
            int highval = std::max(object1, object2);
            int lowval = std::min(object1, object2);
            m_ResemblanceLables.erase(m_ResemblanceLables.begin()+highval);
            m_ResemblanceLables.erase(m_ResemblanceLables.begin()+lowval);

            for ( size_t i = 0; i < m_ResemblanceLables.size(); i++ )
                newResemblanceLables.push_back(m_ResemblanceLables[i]);

            m_ResemblanceLables.clear();
            m_ResemblanceLables = newResemblanceLables;
            m_ResemblanceLables.push_back(newcluster);


            // 3. update the ResemblanceMatix by removing the data at object1 and object2 and
            // adding a new row labled (object1,object2)
            nCurObjects -= 1;  // new matrix will have one less row and col
            CvMat* newResemblanceMatrix = cvCreateMat(nCurObjects, nCurObjects, CV_32FC1);
            CalcResemblanceMatrix(t, nCurObjects, true, newResemblanceMatrix, object1, object2);

            cvReleaseMat(&m_pResemblanceMatrix);
            m_pResemblanceMatrix = newResemblanceMatrix;
        }
    }
    catch (...)
    {
        throw;
    }

    return bRet;
}


void UPGMA::CalcResemblanceMatrix(ResemblanceCoefficientType t, int nObjects, bool bRevise, CvMat* newmatrix, int obj1, int obj2)
{
    if ( bRevise )
    {
        ReviseUPGMACoefficientMatrix( this, nObjects, newmatrix, obj1, obj2 );
    }
    else
    {
        switch (t)
        {
        case BrayCurtisCoefficient:
            CalcBrayCurtisCoefficient(this, nObjects);
            break;
        case CanberraMetricCoefficient:
            CalcCanberraMetricCoefficient(this, nObjects);
            break;
        case CoefficientOfShapeDiff:
            CalcCoefficientOfShapeDiff(this, nObjects);
            break;
        case CorrelationCoefficient:
            CalcCorrelationCoefficient(this, nObjects);
            break;
        case CosineCoefficient:
            CalcCosineCoefficient(this, nObjects);
            break;
        case EuclideanDistanceCoefficient:
            CalcEuclideanDistanceCoefficient(this, nObjects);
            break;
        case MahalanobisDistanceCoefficient:
            CalcMahalanobisDistanceCoefficient(this, nObjects);
            break;
        default:
            throw std::string("CalcDistanceCoefficient - invalid ResemblanceCoefficientType");
        }
    }
}






bool UPGMA::CheckThreshold(double currentThreshold)
{
    return ( currentThreshold <= m_Threshold && m_bIsDisimilarityCoeffcient ) ||
           ( currentThreshold >= m_Threshold && !m_bIsDisimilarityCoeffcient );
}


void UPGMA::Clear()
{
    m_nObjects = 0;
    m_nAttributes = 0;

    if ( m_pDataMatrix )
    {
        cvReleaseMat(&m_pDataMatrix);
        m_pDataMatrix = NULL;
    }

    if ( m_pResemblanceMatrix )
    {
        cvReleaseMat(&m_pResemblanceMatrix);
        m_pResemblanceMatrix = NULL;
    }

    if ( m_pOriginalResemblanceMatrix )
    {
        cvReleaseMat(&m_pOriginalResemblanceMatrix);
        m_pOriginalResemblanceMatrix = NULL;
    }

    if ( m_pCopheneticMatrix )
    {
        cvReleaseMat(&m_pCopheneticMatrix);
        m_pCopheneticMatrix = NULL;
    }

    m_Steps.clear();
    m_ResemblanceLables.clear();
}


void UPGMA::GetStrClusterSteps(std::string& output, bool bPrintAllClusters)
{
    output = "";
    std::stringstream ss;
    for ( size_t i = 0; i < m_Steps.size(); i++ )
    {
        ss << "Step " << i << ": ";
        std::vector<Cluster> clusters = m_Steps[i].clusters;
        for ( size_t j = 0; j < clusters.size(); j++ )
        {
            if ( bPrintAllClusters || clusters[j].bIsNew )
            {
                std::vector<int> objects = clusters[j].objects;
                if ( !objects.empty() )
                {
                    ss << "(";
                    for ( size_t k = 0; k < objects.size()-1; k++ )
                        ss << objects[k] << "[ID:" << personIDMatrix->data.i[objects[k]] << "] ";
                    ss << objects[objects.size()-1] << "[ID:" << personIDMatrix->data.i[objects[objects.size()-1]] << "]";
                    ss << ") distance: " << clusters[j].distance << " ";
                }
            }
        }
        ss << std::endl << std::endl;
    }
    output = ss.str();
}



void UPGMA::GetClustersAtStep( int step, std::string& output )
{
    output = "";
    std::stringstream ss;

    int nSteps = m_Steps.size();

    if ( step > nSteps || step < 0 )
        throw std::string("GetClusterAtStep - Invalid step");

    ss << "Cluster n: (image index|Person ID, ......)" << std::endl;
    ClusterContainer clusters = m_Steps[step].clusters;
    for ( size_t i = 0; i < clusters.size(); i++ )
    {
        std::vector<int> objects = clusters[i].objects;
        size_t nObjects = objects.size();

        ss << "Cluster " << i << ": (";
        for ( size_t j = 0; j < nObjects-1; j++ )
        {
            ss << objects[j] << "|" << personIDMatrix->data.i[objects[j]] << ", ";
        }
        ss << objects[nObjects-1] << "|" << personIDMatrix->data.i[objects[nObjects-1]] << ")" << std::endl;
    }

    output = ss.str();
}


void UPGMA::GetClustersAtClusterCount( int nClusters, std::string& output )
{
    std::cout << "1 ";
    output = "";
    std::stringstream ss;

    size_t step = 0;
    for ( step = 0; step < m_Steps.size() && (size_t)nClusters <= m_Steps[step].clusters.size(); step++ )
    {}

    std::cout << "2 step: " << step;
    ss << "Cluster n: (image index|Person ID, ......)" << std::endl;
    ClusterContainer clusters = m_Steps[step].clusters;
    std::cout << "3 ";
    for ( size_t i = 0; i < clusters.size(); i++ )
    {
        std::vector<int> objects = clusters[i].objects;
        size_t nObjects = objects.size();

        ss << "Cluster " << i << ": (";
        for ( size_t j = 0; j < nObjects-1; j++ )
        {
            ss << objects[j] << "|" << personIDMatrix->data.i[objects[j]] << ", ";
        }
        ss << objects[nObjects-1] << "|" << personIDMatrix->data.i[objects[nObjects-1]] << ")" << std::endl;
    }
    std::cout << "4" << std::endl;

    output = ss.str();
}



UPGMA::~UPGMA()
{
    Clear();

    if ( m_pDatabase )
        delete m_pDatabase;
}




bool UPGMA::LoadTestImages()
{
    bool bRet = true;

    Clear();

    m_nObjects = 5;
    m_nAttributes = 2;
    m_nPeople = 3;

    m_pDataMatrix = cvCreateMat(m_nObjects, m_nAttributes, CV_32FC1);
    m_pDataMatrix->data.fl[0] = 10.0;
    m_pDataMatrix->data.fl[1] = 5.0;
    m_pDataMatrix->data.fl[2] = 20.0;
    m_pDataMatrix->data.fl[3] = 20.0;
    m_pDataMatrix->data.fl[4] = 30.0;
    m_pDataMatrix->data.fl[5] = 10.0;
    m_pDataMatrix->data.fl[6] = 30.0;
    m_pDataMatrix->data.fl[7] = 15.0;
    m_pDataMatrix->data.fl[8] = 5.0;
    m_pDataMatrix->data.fl[9] = 10.0;


    return bRet;
}



bool UPGMA::LoadTestImagesAlt()
{
    bool bRet = true;

    Clear();

    m_nObjects = 4;
    m_nAttributes = 4;
    m_nPeople = 2;

    m_pDataMatrix = cvCreateMat(m_nObjects, m_nAttributes, CV_32FC1);
    m_pDataMatrix->data.fl[0] = 20.0;
    m_pDataMatrix->data.fl[1] = 40.0;
    m_pDataMatrix->data.fl[2] = 25.0;
    m_pDataMatrix->data.fl[3] = 30.0;
    m_pDataMatrix->data.fl[4] = 35.0;
    m_pDataMatrix->data.fl[5] = 55.0;
    m_pDataMatrix->data.fl[6] = 40.0;
    m_pDataMatrix->data.fl[7] = 45.0;
    m_pDataMatrix->data.fl[8] = 40.0;
    m_pDataMatrix->data.fl[9] = 80.0;
    m_pDataMatrix->data.fl[10] = 50.0;
    m_pDataMatrix->data.fl[11] = 60.0;
    m_pDataMatrix->data.fl[12] = 20.0;
    m_pDataMatrix->data.fl[13] = 0.0;
    m_pDataMatrix->data.fl[14] = 15.0;
    m_pDataMatrix->data.fl[15] = 10.0;

    return bRet;
}
