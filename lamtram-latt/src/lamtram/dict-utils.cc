#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <dynet/dict.h>
#include <lamtram/dict-utils.h>
#include <lamtram/string-util.h>
#include <lamtram/macros.h>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>
using namespace std;

namespace lamtram {


vector<string> SplitWords(const std::string & line) {
  std::istringstream in(line);
  std::string word;
  std::vector<std::string> res;
  while(in) {
    in >> word;
    if (!in || word.empty()) break;
    res.push_back(word);
  }
  return res;
}

Sentence ParseAsWords(dynet::Dict & sd, const string& line, bool add_end, string file_type) {
  assert(file_type == "txt" || file_type == "lat");
  if(file_type == "lat"){
    assert(!add_end);
    Sentence ret;
    ret = ParseAsLattice(sd, line, file_type);
    ret.set_lattice(false);
    ret.set_flat(true);
    return ret;
  } else {
    istringstream in(line);
    string word;
    Sentence res;
    while(in) {
      in >> word;
      if (!in || word.empty()) break;
      res.add_word(sd.convert(word));
    }
    if(add_end && (res.size() == 0 || res[0] != 0))
      res.add_word(0);
    res.set_flat(true);
    return res;
  }
}


Sentence ParseAsFeatures(const std::string& line, string file_type) {
  assert(file_type == "feat");
  Sentence res;
  const std::string& normalized_line = RemoveExtraWhitespace(line);
  vector<string> words = Tokenize(normalized_line, ";");
  if(std::string::npos != normalized_line.find_first_not_of("0123456789 e+-.;")){
      THROW_ERROR("Malformed features vector line");
  }
  for(int i=0; i<words.size(); i++){
    vector<string> feats = Tokenize(words[i], " ");
    vector<dynet::real> featsReal = ConvertStringVectorToFloatVector(feats);
    res.add_feature(featsReal);
  }
  res.set_flat(true);
  return res;
}

Sentence ParseAsLattice(dynet::Dict & sd, const std::string& line, string file_type) {
  assert(file_type == "txt" || file_type == "lat");
  if(file_type == "txt"){
    Sentence ret;
    ret = ParseAsWords(sd, line, false, file_type);
    ret.set_lattice(true);
    ret.set_flat(true);
    for(int i=0; i<ret.size()-1; i++){
      if(i<0 || i+1>=ret.size()) THROW_ERROR("Attempting to add edges between non-existing nodes for line: " << line);
      ret.add_lattice_edge(i, i+1);
    }
    for(int i=0; i<ret.size(); i++){
      ret.add_scores({0.0, 0.0, 0.0}); // these are log probs
    }
    return ret;
  } else {
    std::string normalized_line = RemoveExtraWhitespace(line);
    if(normalized_line.substr(0,1)!="[" || normalized_line.substr(normalized_line.size()-1,1)!="]")
      THROW_ERROR("improperly formatted lattice file");
    if(normalized_line.substr(0,2)=="[]")
      THROW_ERROR("lattice illegal because it does not contain any words");

    string lstr = normalized_line.substr(2, normalized_line.size()-4);

    vector<string> nodeEdgeStrs = Tokenize(lstr, ")],[(");
    string concatNodeStr;
    string concatEdgeStr;
    if(lstr.substr(lstr.size()-2,2) == "],"){
      concatNodeStr = nodeEdgeStrs[0];
      concatEdgeStr = "";
    } else {
      if(nodeEdgeStrs.size()!=2) THROW_ERROR("improperly formatted lattice file");
      concatNodeStr = nodeEdgeStrs[0];
      concatEdgeStr = nodeEdgeStrs[1];
    }

    vector<string> nodeStrs = Tokenize(concatNodeStr, "),(");
    vector<string> edgeStrs = Tokenize(concatEdgeStr, "),(");

    if(nodeStrs.size()<=1) THROW_ERROR("lattice illegal because it contains less than 2 nodes");

    Sentence res;
    res.set_lattice(true);
    res.set_flat(false);

    for(int i=0; i<nodeStrs.size(); i++){
      vector<string> wordScoreStr = Tokenize(nodeStrs[i], ",");
      string word = wordScoreStr[0].substr(1, wordScoreStr[0].size()-2);
      float fwdScore = stod(wordScoreStr[1]);
      float marScore = stod(wordScoreStr[2]);
      float bwdScore = stod(wordScoreStr[3]);
      WordId wordId;
      wordId = sd.convert(word);
      res.add_word(wordId);
      res.add_scores({fwdScore, marScore, bwdScore});
    }

    for(int i=0; i<edgeStrs.size(); i++){
      vector<string> fromToStr = Tokenize(edgeStrs[i], ",");
      int from = stoi(fromToStr[0]);
      int to = stoi(fromToStr[1]);
      res.add_lattice_edge(from, to);
    }
    return res;
  }

}

Sentence ParseWords(dynet::Dict & sd, const std::vector<std::string>& line, bool add_end) {
  Sentence res;
  for(const std::string & word : line)
    res.add_word(sd.convert(word));
  if(add_end && (res.size() == 0 || *res.get_word_ids().rbegin() != 0))
    res.add_word(0);
  return res;
}

std::string PrintWords(dynet::Dict & sd, const Sentence & sent) {
  ostringstream oss;
  if(sent.size())
    oss << sd.convert(sent[0]);
  for(size_t i = 1; i < sent.size(); i++)
    oss << ' ' << sd.convert(sent[i]);
  return oss.str();
}
std::string PrintWords(const std::vector<std::string> & sent) {
  ostringstream oss;
  if(sent.size())
    oss << sent[0];
  for(size_t i = 1; i < sent.size(); i++)
    oss << ' ' << sent[i];
  return oss.str();
}

vector<string> ConvertWords(dynet::Dict & sd, const Sentence & sent, bool sentend) {
  vector<string> ret;
  for(WordId wid : sent.get_word_ids()) {
    if(sentend || wid != 0)
      ret.push_back(sd.convert(wid));
  }
  return ret;
}

void WriteDict(const dynet::Dict & dict, const std::string & file) {
  ofstream out(file);
  if(!out) THROW_ERROR("Could not open file: " << file);
  WriteDict(dict, out);
}
void WriteDict(const dynet::Dict & dict, std::ostream & out) {
  out << "dict_v001" << '\n';
  for(const auto & str : dict.get_words())
    out << str << '\n';
  out << endl;
}
dynet::Dict* ReadDict(const std::string & file) {
  ifstream in(file);
  if(!in) THROW_ERROR("Could not open file: " << file);
  return ReadDict(in);
}
dynet::Dict* ReadDict(std::istream & in) {
  dynet::Dict* dict = new dynet::Dict;
  string line;
  if(!getline(in, line) || line != "dict_v001")
    THROW_ERROR("Expecting dictionary version dict_v001, but got: " << line);
  while(getline(in, line)) {
    if(line == "") break;
    dict->convert(line);
  }
  bool has_unk = dict->convert("<unk>") == 1;
  dict->freeze();
  if(has_unk)
    dict->set_unk("<unk>");
  return dict;
}
dynet::Dict * CreateNewDict(bool add_tokens) {
  dynet::Dict * ret = new dynet::Dict;
  if(add_tokens) {
    ret->convert("<s>");
    ret->convert("<unk>");
  }
  return ret;
}

}
