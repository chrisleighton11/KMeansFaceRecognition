#ifndef IMAGESTRUCT_H
#define IMAGESTRUCT_H


#include "Utilities.h"

// structure to store ImageFile contents
struct Image
{
    int            m_ID;
    std::string    m_PersonName;
    std::string    m_ImageName;
    IplImage*      m_Image;

    Image()
    {
        m_ID = 0;
        m_PersonName = "";
        m_ImageName = "";
        m_Image = NULL;
    }

    Image(const char* buffer)
    {
        std::string stuff(buffer);

        // LOOK!!!!! this assumes that the line is space delimited
        // needs improvement if to become production code like some error checking


        // get the id
        size_t pos = stuff.find_first_of(' ',0);
        if ( pos == std::string::npos )
            throw std::string("Image: Bad line");
        m_ID = atoi(stuff.substr(0,pos).c_str());

        // get persons name
        size_t pos2 = stuff.find_first_of(' ',pos+1);
        if ( pos2 == std::string::npos )
            throw std::string("Image: Bad line");
        m_PersonName = stuff.substr(pos+1,pos2-pos-1);

        // get the image name
        m_ImageName = stuff.substr(pos2+1);
    }
};

#endif
