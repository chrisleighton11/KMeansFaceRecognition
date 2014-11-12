#ifndef HTMLHELPER_H
#define HTMLHELPER_H

#include "Utilities.h"
#include <iostream>
#include <string>

/*
	HTMLHelper
	Defines functions to create HTML

	Chris Leighton
	CS6420 Class Project
	March 3rd 2011
*/


std::string GetHeader();

std::string GetTitle(const char* title);

std::string GetBr(const char* spaces);

std::string GetText(const char *text, const char* sz, const char* spaces);

std::string GetImageTag(const char* imagename, const char* width, const char* height, const char* spaces);



std::string GetClosingTags();


#endif
