#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <boost/functional/hash.hpp>

#include <dynet/tensor.h>

namespace lamtram {

typedef int WordId;
//typedef std::vector<WordId> Sentence;
typedef std::vector<std::vector<float> > Alignment;
typedef std::unordered_map<std::string, std::pair<std::string, float> > Mapping;



class Sentence {
private:
  std::vector<WordId> word_ids;                 // discrete word IDs
  std::vector<std::vector<dynet::real>> features;    // word features, (e.g. word embeddings or sound features)
  std::vector<std::vector<float>> scores;                   // uncertainty scores etc. assigned to the words
  std::vector<std::vector<int>> successors;    // list of successor nodes for each word, for graph-based sentence representation
  std::vector<std::vector<int>> predecessors;  // list of successor nodes for each word, for graph-based sentence representation
  bool marked_as_lattice, marked_as_flat;

public:
  Sentence() : word_ids(), marked_as_lattice(false) {}
  Sentence(std::size_t count) : word_ids(std::vector<WordId>(count, 0)), marked_as_lattice(false), marked_as_flat(false) {}
  Sentence(std::vector<WordId> wordIds) : word_ids(wordIds), marked_as_lattice(false) , marked_as_flat(false) {}
  Sentence(std::size_t count, const WordId& value) : word_ids(std::vector<WordId>(count, value)), marked_as_lattice(false) {}

  const bool operator< (const Sentence& rhs) const;
  bool operator==(const Sentence &other) const;

  WordId& operator[] (std::size_t position) {return word_ids[position];}
  const WordId operator[] (std::size_t position) const {return word_ids[position];}
  Sentence& operator=(const std::vector<WordId> &rhs) {this->word_ids = rhs; return *this;}

  std::size_t size() const;
  void clear();
  void add_word( const WordId& value );


  const std::vector<WordId> get_word_ids() const {return word_ids;}
  WordId get_last_wordid() const { return *word_ids.rbegin(); }
  void resize_words(std::size_t new_size) { word_ids.resize(new_size); }
  void insert_word_begin(const WordId& val) { word_ids.insert(word_ids.begin(), val); }
  void erase_first(){ word_ids.erase(word_ids.begin()); }

  const std::size_t get_hash() const;

  void set_lattice(bool val);
  void set_flat(bool val);
  bool is_lattice() const;
  bool is_flat() const;
  bool is_feature_represented() const;
  bool has_scores() const;
  const std::vector<dynet::real> get_feature(int pos) const;
  unsigned int get_feature_dim() const;
  const std::vector<dynet::real> get_zero_feature() const;


  const std::vector<int> get_predecessors(int nodeIndex) const;
  const std::vector<int> get_successors(int nodeIndex) const;

  const float get_score_as_prob(int pos, int scoreIndex) const;
  const std::vector<float> get_multiple_scores_as_prob(std::vector<int> positions, int scoreIndex) const;


  void add_scores( std::vector<float> score );
  void add_feature( std::vector<dynet::real> feature );
  void add_lattice_edge( int from, int to);

};



}


namespace std
{
  template<> struct hash<lamtram::Sentence>
  {
    typedef lamtram::Sentence argument_type;
    typedef std::size_t result_type;
    result_type operator()(argument_type const& s) const
    {
      return s.get_hash();
    }
  };
}
