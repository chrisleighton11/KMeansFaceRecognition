#ifndef UPGMA_H
#define UPGMA_H

#include "Utilities.h"
#include "Database.h"
#include "ResemblanceCoefficient.h"
#include "Cluster.h"

class UPGMA
{
public:
    UPGMA( const char* databaseName );
    ~UPGMA();

    // Load DataMatrix
    bool LoadImages();        // Load DataMatrix with original images
    bool LoadTestImages();    // Test method with known resutls from Clustering book
    bool LoadTestImagesAlt();
    bool LoadReducedImages(); // Load DataMatrix with projected values after PCA

    // Run the clustering algorithm
    bool DoCluster( ResemblanceCoefficientType t = EuclideanDistanceCoefficient );


    CvMat* GetDataMatrix() { return m_pDataMatrix; }
    CvMat* GetResemblanceMatrix() { return m_pResemblanceMatrix; }
    CvMat* GetOriginalResemblanceMatrix() { return m_pOriginalResemblanceMatrix; }
    ClusterContainer& GetResemblanceLables() { return m_ResemblanceLables; }
    int    GetnObjects() { return m_nObjects; }
    int    GetnPeople() { return m_nPeople; }
    int    GetnAttributes() { return m_nAttributes; }
    bool   IsDisimilarityCoeffcient() { return m_bIsDisimilarityCoeffcient; }

    void GetStrClusterSteps(std::string& output, bool bPrintAllClusters);
    void GetClustersAtStep( int step, std::string& output );
    void GetClustersAtClusterCount( int nClusters, std::string& output );


private:
    void Clear();
    void CalcResemblanceMatrix(ResemblanceCoefficientType t, int nObjects, bool bRevise = 0, CvMat* newmatrix = NULL, int obj1 = 0, int obj2 = 0);
    bool CheckThreshold(double currentThreshold);

    CvMat*      m_pDataMatrix;  // each row corresponds to an image, columns are the attributes
                                // each row index can be translated to person id with the databases
                                // personIDMatrix
                                // the cvMat will be m_nObjects by nAttributes
    int         m_nObjects;
    int         m_nPeople;
    int         m_nAttributes;

    CvMat*                  m_pResemblanceMatrix; // m_nObjects by m_nObjects mat to store resemblance coefficients
    CvMat*                  m_pOriginalResemblanceMatrix; // original used for recalculating new resemblance matrix
    ClusterContainer        m_ResemblanceLables;  // each row of the Resemblance matrix really represents a cluster lable
                                                  // not an object
    bool        m_bIsDisimilarityCoeffcient;  // true if resemblance coefficient is the type dissimilarity

    CvMat*      m_pCopheneticMatrix;   // stores distances from clustering algorithm
    std::vector<Cluster_step> m_Steps;

    Database*   m_pDatabase;
    double      m_Threshold;

};




#endif

