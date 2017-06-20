#pragma once

#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <lamtram/macros.h>

namespace lamtram {

inline std::vector<std::string> TokenizeWildcarded(const std::string & str, const std::vector<std::string> & wildcards, const std::string & delimiter) {
  std::vector<std::string> ret;
  auto end = str.find("WILD");
  if(end != std::string::npos) {
    std::string left = str.substr(0, end);
    std::string right = str.substr(end+4);
    for(size_t i = 0; i < wildcards.size(); i++)
      ret.push_back(left+wildcards[i]+right);
  } else {
    boost::split(ret, str, boost::is_any_of(delimiter));
  }
  return ret;
}

inline std::vector<std::string> Tokenize(const char *str, char c = ' ') {  
  std::vector<std::string> result;
  if(*str == 0) return result;
  while(1) {
    const char *begin = str;
    while(*str != c && *str)
      str++;
    result.push_back(std::string(begin, str));
    if(0 == *str++)
      break;
  }
  return result;
}
inline std::vector<std::string> Tokenize(const std::string &str, char c = ' ') {
  return Tokenize(str.c_str(), c);
}
inline std::vector<std::string> Tokenize(const std::string & str, const std::string & delim) {
  std::vector<std::string> vec;
  if(str == "") return vec;
  size_t loc, prev = 0;
  while((loc = str.find(delim, prev)) != std::string::npos) {
    vec.push_back(str.substr(prev, loc-prev));
    prev = loc + delim.length();
  }
  vec.push_back(str.substr(prev, str.size()-prev));
  return vec;
}
inline bool BothAreSpaces(char lhs, char rhs) { return (lhs == rhs) && (lhs == ' '); }
inline std::string RemoveExtraWhitespace(const std::string & str) {

  std::string normalized_str = str;
  std::string::iterator new_end = std::unique(normalized_str.begin(), normalized_str.end(), BothAreSpaces);
  normalized_str.erase(new_end, normalized_str.end());

  boost::replace_all(normalized_str, ", ", ",");
  boost::replace_all(normalized_str, " ,", ",");
  boost::replace_all(normalized_str, "; ", ";");
  boost::replace_all(normalized_str, " ;", ";");
  boost::replace_all(normalized_str, " [", "[");
  boost::replace_all(normalized_str, " ]", "]");
  boost::replace_all(normalized_str, "[ ", "[");
  boost::replace_all(normalized_str, "] ", "]");
  boost::replace_all(normalized_str, " )", ")");
  boost::replace_all(normalized_str, " (", "(");
  boost::replace_all(normalized_str, ") ", ")");
  boost::replace_all(normalized_str, "( ", "(");

  return normalized_str;
}
inline std::string FirstToken(const std::string & str, char c = ' ') {
  const char *end = &str[0];
  while(*end != 0 && *end != c)
    end++;
  return std::string(&str[0], end);
}

inline std::string EscapeQuotes(std::string ret) {
  boost::replace_all(ret, "\\", "\\\\");
  boost::replace_all(ret, "\"", "\\\"");
  return ret;
}

inline std::vector<float> ConvertStringVectorToFloatVector(const std::vector<std::string>& stringVector){
  std::vector<float> doubleVector(stringVector.size());
  std::transform(stringVector.begin(), stringVector.end(), doubleVector.begin(), [](const std::string& val) {
    return stof(val);
  });
  return doubleVector;
}

// split file specifier "lat:mytrain-data.en" into filetype "lat" and filename "mytrain-data.en"
inline void GetFileNameAndType(const std::string & file_type_name, std::string & filename, std::string & filetype, const std::string & default_type){
  std::vector<std::string> filename_spl = Tokenize(file_type_name, "=");
  if(filename_spl.size()>1){
    if(filename_spl.size()>2) {
	THROW_ERROR("Too many colons in file specifier: " << file_type_name);
    } else {
      filename = filename_spl[1];
      filetype = filename_spl[0];
    }
  } else {
    filename = file_type_name;
    filetype = default_type;
  }
}


}  // end namespace
