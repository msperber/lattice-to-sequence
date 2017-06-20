/*
 * sentence.cc
 *
 *  Created on: Aug 31, 2016
 *      Author: matthias
 */

#include <algorithm>
#include <lamtram/sentence.h>
#include <lamtram/macros.h>
#include <math.h>

namespace lamtram {


  void Sentence::clear() {
    word_ids.clear();
    scores.clear();
    successors.clear();
    predecessors.clear();
    features.clear();
  }

  std::size_t Sentence::size() const {
    return std::max(word_ids.size(),
	       std::max(scores.size(),
		   std::max(successors.size(),
		       std::max(predecessors.size(), features.size()))));
  }

  void Sentence::add_word( const WordId& value ) {
    word_ids.push_back(value);
  }


  const bool Sentence::operator< (const Sentence& rhs) const {
    if(word_ids==rhs.word_ids){
      if(successors==rhs.successors){
	if(predecessors==rhs.predecessors){
	  return scores < rhs.scores;
	} else {
	  return predecessors < rhs.predecessors;
	}
      } else {
	return successors < rhs.successors;
      }
    } else {
      return word_ids < rhs.word_ids;
    }
  }

  bool Sentence::operator==(const Sentence &other) const {
    return word_ids == other.word_ids && successors==other.successors && predecessors==other.predecessors && scores==other.scores;
  }

  const std::size_t Sentence::get_hash() const {
    std::size_t const h1 ( boost::hash<std::vector<lamtram::WordId>>()(get_word_ids()) );
    return h1; // or use boost::hash_combine
    //          result_type const h1 ( std::hash<std::string>()(s.first_name) );
    //          result_type const h2 ( std::hash<std::string>()(s.last_name) );
    //          return h1 ^ (h2 << 1); // or use boost::hash_combine
  }


  const std::vector<int> Sentence::get_predecessors(int nodeIndex) const {
    assert(nodeIndex < size());
    if(is_lattice()){
      if(nodeIndex >= predecessors.size())
	return std::vector<int>();
      else
	return predecessors[nodeIndex];
    } else {
      if(nodeIndex<=0){
	  return std::vector<int>();
      } else {
	  return std::vector<int>(1,nodeIndex-1);
      }
    }
  }
  const std::vector<int> Sentence::get_successors(int nodeIndex) const {
    assert(nodeIndex < size());
    if(is_lattice()){
      if(nodeIndex >= successors.size())
	return std::vector<int>();
      else
	return successors[nodeIndex];
    } else {
      if(nodeIndex>=size()-1){
	  return std::vector<int>();
      } else {
	  return std::vector<int>(1,nodeIndex+1);
      }
    }
  }

  const float Sentence::get_score_as_prob(int pos, int scoreIndex) const {
    if(scores[pos][scoreIndex] > 0.001){ // give it some slack
	THROW_ERROR("encountered what is assumed to be a log-prob, but turns out to be > 0.001: " << scores[pos][scoreIndex] );
    }
    return exp(scores[pos][scoreIndex]);
  }


  const std::vector<float> Sentence::get_multiple_scores_as_prob(std::vector<int> positions, int scoreIndex) const {
    std::vector<float> probs;
    for(std::vector<int>::iterator it = positions.begin(); it != positions.end(); ++it) {
      float logprob = 0.f;
      if( *it >= 0 && *it < size() ) logprob = scores[*it][scoreIndex];
      if(logprob > 0.001){ // give it some slack
	THROW_ERROR("encountered what is assumed to be a log-prob, but turns out to be > 0.001: " << logprob );
      }
      probs.push_back(exp(logprob));
    }
    return probs;
  }

  bool Sentence::has_scores() const {
    return scores.size() > 0;
  }

  void Sentence::add_scores( std::vector<float> score ) {
    scores.push_back(score);
  }

  void Sentence::add_feature( std::vector<float> feature ) {
    if(features.size()>0 && features[0].size() != feature.size())
      THROW_ERROR("All feature vectors in a sentence must have consistent dimensions!");
    features.push_back(feature );
  }

  void Sentence::set_lattice(bool val){
    marked_as_lattice = val;
  }

  void Sentence::set_flat(bool val){
    marked_as_flat = val;
  }

  bool Sentence::is_lattice() const {
    return marked_as_lattice;
  }

  bool Sentence::is_flat() const {
    return marked_as_flat;
  }

  bool Sentence::is_feature_represented() const {
    return features.size() > 0;
  }

  const std::vector<dynet::real> Sentence::get_feature(int pos) const {
    return features[pos];
  }

  unsigned int Sentence::get_feature_dim() const {
    return features[0].size();
  }
  const std::vector<dynet::real> Sentence::get_zero_feature() const{
    return std::vector<dynet::real>(get_feature_dim());
  }

  void Sentence::add_lattice_edge( int from, int to ){
    assert(is_lattice());
    if(from<0 || from>=size() || to<0 || to>=size())
      THROW_ERROR("Attempting to add edges between non-existing nodes");

    while(successors.size() <= std::max(from, to)){
      successors.push_back(std::vector<int>());
    }
    while(predecessors.size() <= std::max(from, to)){
      predecessors.push_back(std::vector<int>());
    }
    successors[from].push_back(to);
    predecessors[to].push_back(from);
  }


}
