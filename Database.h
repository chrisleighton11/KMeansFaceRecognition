#ifndef DATABASE_H
#define DATABASE_H

#include "Utilities.h"
#include "ImageStruct.h"


extern IplImage**  imageArray;
extern IplImage**  eigenVectorArray;
extern IplImage*   averageImage;     // Average image of all training images
extern CvMat*      personIDMatrix;   // matrix to store person ids
extern CvMat*      eigenValueMatrix; // matrix to store Eigen values
extern CvMat*      projectedFaceMatrix; // matrix to store projected faces



class Database
{
public:

    typedef std::vector<std::string> NameVec;
    typedef std::vector<Image>       ImageVec;

    Database();
    ~Database();

    bool Write( const std::string& databaseName );
    bool Read( const std::string& databaseName );

    bool ValidateData();
    void ClearExternalData();

    void SetnImages( int n ) { m_nImages = n; }
    int  GetnImages() { return m_nImages; }

    void SetnPeople( int n ) { m_nPeople = n; }
    int  GetnPeople() { return m_nPeople; }

    void SetnEigenVals( int n ) { m_nEigenVals = n; };
    int  GetnEigenVals() { return m_nEigenVals; }

    void SetEuclideanThreshold( double t ) { m_EuclideanThreshold = t; }
    double GetEuclideanThreshold() { return m_EuclideanThreshold; }

    void SetMahalanobisThreshold( double t ) { m_MahalanobisThreshold = t; }
    double GetMahalanobisThreshold() { return m_MahalanobisThreshold; }

    void SetNames( NameVec& names ) { m_Names = names; }
    NameVec& GetNames() { return m_Names; }

    void SetImageVec( ImageVec& images ) { m_ImageVec = images; }
    ImageVec& GetImageVec() { return m_ImageVec; }


private:
    CvFileStorage*              m_Storage;

    // data
    int                         m_nImages;
    int                         m_nPeople;
    int                         m_nEigenVals;

    double                      m_EuclideanThreshold;   		// 1/2 the largest euclidean distance for each projected face
    double                      m_MahalanobisThreshold;	// 1/2 the largest Mahalanobis distance for each projected face

    NameVec                     m_Names;
    ImageVec                    m_ImageVec;



};




#endif
