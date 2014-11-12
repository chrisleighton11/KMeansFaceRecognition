#ifndef TRAINING_H
#define TRAINING_H

/*
   Training.h
   Description:   defines all of the training classes and methods
   Author:        Chris Leighton
   Date:          Feb 16th 2010

*/

#include <fstream>
#include <vector>

#include "Utilities.h"
#include "Database.h"
#include "ImageStruct.h"

void Train(const char* imagelist, const char* database, std::string& resultdir);

class Trainer
{
public:
    Trainer(const char* imagelist, const char* database);
    ~Trainer();

    int LoadImages();
    void CreateSubspace();
    void ProjectOntoSubSpace();
    void StoreData();
    void GenResults(std::string& resultsdir);
    void CalculateThresholds();
    void MakeDatabase();

private:
    std::string             m_ImageFile;      // list of images of faces and thier names
    std::string             m_DatabaseFile;   // where to put the results

    Database*                m_pDatabase;
};





#endif


