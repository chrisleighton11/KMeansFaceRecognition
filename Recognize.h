#ifndef RECOGNIZE_H
#define RECOGNIZE_H

/*
   Recognize.h
   Description:   defines all of the classes and methods for the recognition part
   Author:        Chris Leighton
   Date:          Feb 20th 2010

*/

#include <vector>

#include "Utilities.h"
#include "Database.h"


std::string Recognize(const char* image, const char* database, double& distance, std::string& resultsdir, int& idFound, Database* db = NULL, bool bCheckDistance = true);



class Recognizer
{
public:
    Recognizer(const char* imagename, const char* databasename);
    Recognizer( Database* db, const char* imagename, const char* databasename);
    ~Recognizer();


    bool        LoadTrainingDatabase();
    std::string FindFace( int faceNum, double& distance, int& idFound, bool bCheckDistance );
    int         EuclideanDistance( float* projectedTestFace, double& distance );
    int         MahalanobisDistance( float* projectedTestFace, double& distance );

    void	    GenResults(std::string& resultsdir);

private:
    const char*             m_DatabaseName;
    Database*               m_pDatabase;

    const char*             m_SearchImageName;   // name if search image
    IplImage*               m_FaceImage;         // original image with faces to find, could be more than one
    IplImage**              m_FacesToFind;       // array of Faces to find
    int                     m_nFacesToFind;

    // results
    int                     m_IDFound;
    double                  m_DistanceFound;
    std::string             m_PersonFound;     // if we din't find a person it remains ""

    bool                    m_bDeleteDb;

};




#endif

