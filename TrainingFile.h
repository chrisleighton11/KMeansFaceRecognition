#ifndef TRAININGFILE_H
#define TRAININGFILE_H

/* TrainingFile - provides methods to create training file for recognizer
   Chris Leighton
   March 2nd 2011
*/

#include "Utilities.h"
#include <vector>
#include <map>

struct TE
{
    int         id;
    std::string name;
    std::string filename;
};

class TrainingFile
{
public:
    TrainingFile( const std::string& filename ) : m_fileName(filename), m_BaseDir(""), m_lastid(0)  {}

    void SetBaseDir(std::string& base)
    {
        m_BaseDir = base;
    }

    void addEntry( const std::string& name, const std::string& filename );

    bool GenFile();


private:
    std::string		m_fileName;
    std::string     m_BaseDir;

    std::vector<TE>      m_entries;
    std::map<std::string,int>   m_NameToIdMap;

    int                     m_lastid;


};


#endif

