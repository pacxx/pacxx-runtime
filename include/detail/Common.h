//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_COMMON_H
#define PACXX_V2_COMMON_H

namespace pacxx{
    namespace common
    {
        // reads a environment variable
        std::string GetEnv(const std::string &var);

        // extracts the filename from a filepath
        std::string get_file_from_filepath(std::string path);

        // replaces as substring in the subject string
        std::string replace_substring(std::string subject,
                                      const std::string &search,
                                      const std::string &replace);
    }
}

#endif //PACXX_V2_COMMON_H
