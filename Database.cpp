#include "Database.h"
#include "PreProcess.h"


IplImage**  imageArray;
IplImage**  eigenVectorArray;
IplImage*   averageImage;     // Average image of all training images
CvMat*      personIDMatrix;   // matrix to store person ids
CvMat*      eigenValueMatrix; // matrix to store Eigen values
CvMat*      projectedFaceMatrix; // matrix to store projected faces

Database::Database() : m_Storage(NULL), m_nImages(0), m_nPeople(0), m_nEigenVals(0), m_EuclideanThreshold(0.0), m_MahalanobisThreshold(0.0)
{
    imageArray = NULL;
    eigenVectorArray = NULL;
    averageImage = NULL;
    personIDMatrix = NULL;
    eigenValueMatrix = NULL;
    projectedFaceMatrix = NULL;
}


Database::~Database()
{

    if ( m_Storage )
        cvReleaseFileStorage(&m_Storage);

    ClearExternalData();
}


void Database::ClearExternalData()
{
    if (averageImage)
        cvReleaseImage(&averageImage);
    if (personIDMatrix)
        cvReleaseMat(&personIDMatrix);
    if (eigenValueMatrix)
        cvReleaseMat(&eigenValueMatrix);
    if (projectedFaceMatrix)
        cvReleaseMat(&projectedFaceMatrix);

    if ( imageArray )
    {
        for ( int i = 0; i < m_nImages; i++ )
        {
            if ( imageArray[i] )
                cvReleaseImage(&imageArray[i]);
        }
    }

    if ( eigenVectorArray )
    {
        for ( int i = 0; i < m_nEigenVals; i++ )
        {
            if ( eigenVectorArray[i] )
                cvReleaseImage(&eigenVectorArray[i]);
        }
    }

    imageArray = NULL;
    eigenVectorArray = NULL;
    averageImage = NULL;
    personIDMatrix = NULL;
    eigenValueMatrix = NULL;
    projectedFaceMatrix = NULL;

}

bool Database::Write( const std::string& databaseName )
{
    bool bRet = true;

    if ( !ValidateData() )
        throw std::string("Can note write database - database not valid");

    if ( m_Storage )
    {
        cvReleaseFileStorage(&m_Storage);
        m_Storage = NULL;
    }

    m_Storage = cvOpenFileStorage(databaseName.c_str(), 0, CV_STORAGE_WRITE);

    if ( !m_Storage )
        throw std::string("Database::Write could not open database");

    cvWriteInt( m_Storage, "nImages", m_nImages );


    std::vector<std::string> unique_names;
    std::vector<std::string>::iterator it;
    for ( size_t i = 0; i < m_Names.size(); i++ )
    {
        it = find(unique_names.begin(), unique_names.end(), m_Names[i]);
        if ( it == unique_names.end() )
            unique_names.push_back(m_Names[i]);
    }

    cvWriteInt( m_Storage, "nPeople", unique_names.size() );

    for ( size_t i = 0; i < m_Names.size(); i++ )
    {
        char var[256];
        sprintf(var,"PersonID_%d", (i+1));
        cvWriteString( m_Storage, var, m_Names[i].c_str(), 0 );
    }

    for ( size_t i = 0; i < m_ImageVec.size(); i++ )
    {
        char var[256];
        sprintf(var, "ImageID_%d", i);
        cvWriteString( m_Storage, var, m_ImageVec[i].m_ImageName.c_str(), 0 );
    }

    cvWriteInt( m_Storage, "nEigenVals", m_nEigenVals );
    cvWrite( m_Storage, "PersonIDMatrix", personIDMatrix, cvAttrList(0,0) );
    cvWrite( m_Storage, "EigenValueMatrix", eigenValueMatrix, cvAttrList(0,0) );
    cvWrite( m_Storage, "ProjectedFaceMatrix", projectedFaceMatrix, cvAttrList(0,0) );
    cvWrite( m_Storage, "AverageImage", averageImage, cvAttrList(0,0) );

    // store each eigen vector that we saved off
    for ( int i = 0; i < m_nImages-1; i++ )
    {
        char var[256];
        sprintf(var ,"EigenVector_%d",i);
        cvWrite( m_Storage, var, eigenVectorArray[i], cvAttrList(0,0) );
    }

    // store threshold values
    cvWriteReal( m_Storage, "EuclideanThreshold", m_EuclideanThreshold );
    cvWriteReal( m_Storage, "MahalanobisThreshold", m_MahalanobisThreshold );

    return bRet;
}


bool Database::Read( const std::string& databaseName )
{
    bool bRet = true;

    if ( m_Storage )
    {
        cvReleaseFileStorage(&m_Storage);
        m_Storage = NULL;
    }

    m_Storage = cvOpenFileStorage(databaseName.c_str(), 0, CV_STORAGE_READ);

    if ( !m_Storage )
        throw std::string("Database::Read could not open database");

    m_nImages = cvReadIntByName( m_Storage, 0, "nImages", 0 );
    m_nPeople = cvReadIntByName( m_Storage, 0, "nPeople", 0 );

    personIDMatrix = (CvMat*)cvReadByName( m_Storage, 0, "PersonIDMatrix", 0 );

    // read person names and original image names
    for ( int i = 0; i < m_nImages; i++ )
    {
        Image img;
        std::string tempName;
        char varname[256];
        sprintf(varname,"PersonID_%d",i+1);
        tempName = cvReadStringByName( m_Storage, 0, varname, 0 );
        m_Names.push_back(tempName);
        img.m_PersonName = tempName;

        sprintf(varname, "ImageID_%d", i);
        tempName = cvReadStringByName( m_Storage, 0, varname, 0 );

        IplImage* image = cvLoadImage(tempName.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        if (!image)
            throw std::string("Database::Read could not find original image");

        IplImage* newimage = NULL;
        PreProcess(image, &newimage);

        img.m_ID = personIDMatrix->data.i[i];
        img.m_Image = newimage;
        img.m_ImageName = tempName;


        m_ImageVec.push_back(img);
    }


    m_nEigenVals = cvReadIntByName( m_Storage, 0, "nEigenVals", 0 );
    averageImage = (IplImage*)cvReadByName( m_Storage, 0, "AverageImage", 0 );
    eigenValueMatrix = (CvMat*)cvReadByName( m_Storage, 0, "EigenValueMatrix", 0 );
    projectedFaceMatrix = (CvMat*)cvReadByName( m_Storage, 0, "ProjectedFaceMatrix", 0 );

    eigenVectorArray = (IplImage**)cvAlloc(m_nImages*sizeof(IplImage*));
    for ( int i = 0; i < m_nEigenVals; i++ )
    {
        char var[256];
        sprintf(var ,"EigenVector_%d",i);
        eigenVectorArray[i] = (IplImage*)cvReadByName(m_Storage, 0, var, 0);
    }

    m_EuclideanThreshold = cvReadRealByName( m_Storage, 0, "EuclideanThreshold", 0 );
    m_MahalanobisThreshold = cvReadRealByName (m_Storage, 0, "MahalanobisThreshold", 0 );

    return bRet;
}



bool Database::ValidateData()
{
    if ( m_nImages <= 0         ||
         m_nEigenVals <= 0      ||
         averageImage == NULL ||
         m_EuclideanThreshold < 0
         )
         return false;

    return true;


}
