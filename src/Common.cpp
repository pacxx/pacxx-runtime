//
// Created by mhaidl on 29/05/16.
//

#include <fstream>
#include <iostream>
#include <cstdlib>


#include "detail/Common.h"

namespace pacxx{
    namespace common
    {
        std::string GetEnv(const std::string &var) {
            const char *val = ::getenv(var.c_str());
            if (val == 0) {
                return "";
            } else {
                return val;
            }
        }

        std::string get_file_from_filepath(std::string path) {

#ifndef __WIN32__
            std::string delim("/");
#else
            std::string delim("\\");
#endif
            std::string filename;

            size_t pos = path.find_last_of(delim);
            if (pos != std::string::npos)
                filename.assign(path.begin() + pos + 1, path.end());
            else
                filename = path;

            return filename;
        }

        std::string replace_substring(std::string subject,
                                             const std::string &search,
                                             const std::string &replace) {
            size_t pos = 0;
            while ((pos = subject.find(search, pos)) != std::string::npos) {
                subject.replace(pos, search.length(), replace);
                pos += replace.length();
            }
            return subject;
        }

    }
}
