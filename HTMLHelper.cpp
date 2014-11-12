#include "HTMLHelper.h"
#include <sstream>

using namespace std;

std::string GetHeader()
{
    stringstream ss;

    ss << "<?xml version = \"1.0\" encoding = \"utf-8\"?>" << endl;
    ss << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.1//EN\"" << endl;
    ss << "\"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd\">" << endl;

    ss << "<html>" << endl;
    ss << " <head>" << endl;

    return ss.str();
}


std::string GetTitle(const char* title)
{
    stringstream ss;
    ss << "  <title>" << title << "</title>" << endl;
    ss << " </head>" << endl;

    return ss.str();
}


std::string GetBr(const char* spaces)
{
    stringstream ss;
    ss << spaces << "<br>" << endl;

    return ss.str();
}


std::string GetText( const char* text, const char* sz, const char* spaces )
{
    stringstream ss;
    ss << spaces << "<" << sz << ">" << text << "</" << sz << ">" << endl;

    return ss.str();
}



std::string GetImageTag( const char* imagename, const char* width, const char* height, const char* spaces )
{
    stringstream ss;
    ss << spaces << "<img width=" << width << " height=" << height << " src=\"" << imagename << "\"/>" << endl;

    return ss.str();
}




std::string GetClosingTags()
{
    stringstream ss;

    ss << " </body>" << endl;
    ss << "</html>" << endl;

    return ss.str();
}
