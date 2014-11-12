#include "TrainingFile.h"
#include <algorithm>
#include <cctype>
#include <fstream>


using namespace std;



void TrainingFile::addEntry( const std::string& name, const std::string& filename )
{
    TE entry;
    entry.name = name;
    entry.filename = m_BaseDir + filename;

    // see if the name already exists in the entries
    map<string,int>::iterator it;
    if ( (it = m_NameToIdMap.find(name)) != m_NameToIdMap.end() )
        entry.id = it->second;
    else
    {
        entry.id = ++m_lastid;
        m_NameToIdMap[name] = m_lastid;
    }

    m_entries.push_back(entry);
}



bool TrainingFile::GenFile()
{
    bool bRet = true;
    ofstream out(m_fileName.c_str());

    if ( !out.is_open() )
        throw std::string("TrainingFile could not open file");

    for ( size_t i = 0; i < m_entries.size(); i++ )
    {
        out << m_entries[i].id << " " << m_entries[i].name << " " << m_entries[i].filename << endl;
    }

    out.close();
    return bRet;
}


