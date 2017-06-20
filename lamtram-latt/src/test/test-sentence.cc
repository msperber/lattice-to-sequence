#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/sentence.h>
#include <lamtram/dict-utils.h>
#include <dynet/dict.h>
#include <sstream>
#include <stdexcept>


using namespace std;
using namespace lamtram;
namespace utf = boost::unit_test;

// git test

BOOST_AUTO_TEST_SUITE(sentence)

void TestProperWordSequence(Sentence sent, vector<string> desiredWords, DictPtr vocab, bool checkEdges = true){
  BOOST_CHECK_EQUAL(sent.size(), desiredWords.size());
  for(int i=0; i<desiredWords.size(); i++){
    int act = sent[i];
    int des = vocab->convert(desiredWords[i]);
    BOOST_CHECK_EQUAL(act, des);
  }
  if(checkEdges){
    for(int i=0; i<desiredWords.size()-1; i++){
	BOOST_CHECK_EQUAL(sent.get_successors(i).size(), 1);
	BOOST_CHECK_EQUAL(sent.get_successors(i)[0], i+1);
    }
    for(int i=1; i<desiredWords.size(); i++){
	BOOST_CHECK_EQUAL(sent.get_predecessors(i).size(), 1);
	BOOST_CHECK_EQUAL(sent.get_predecessors(i)[0], i-1);
    }
    BOOST_CHECK_EQUAL(sent.get_predecessors(0).size(), 0);
    BOOST_CHECK_EQUAL(sent.get_successors(sent.size()-1).size(), 0);
  }
}

void TestNumEdges(Sentence sent, std::vector<int> numEdges){
  BOOST_CHECK_EQUAL(numEdges.size(), sent.size());
  for(int i=0; i<sent.size(); i++){
    BOOST_CHECK_EQUAL(sent.get_predecessors(i).size(), numEdges[i]);
    BOOST_CHECK_EQUAL(sent.get_successors(sent.size()-i-1).size(), numEdges[i]);

  }
}

BOOST_AUTO_TEST_CASE(TestReadWordsAsWords) {
  DictPtr vocab(CreateNewDict());
  Sentence sent1 = ParseAsWords(*vocab, "a b", false);
  BOOST_CHECK_EQUAL(sent1.is_lattice(), false);
  TestProperWordSequence(sent1, {"a", "b"}, vocab);
  Sentence sent2 = ParseAsWords(*vocab, "a b", true);
  BOOST_CHECK_EQUAL(sent2.is_lattice(), false);
  TestProperWordSequence(sent2, {"a", "b", "<s>"}, vocab);
}

BOOST_AUTO_TEST_CASE(TestReadLatticeAsWords) {
  DictPtr vocab(CreateNewDict());
  string lat1 = "[('a', 0), ('b', -0.1)],[]";
  Sentence sent1 = ParseAsWords(*vocab, lat1, false, "lat");
  BOOST_CHECK_EQUAL(sent1.is_lattice(), false);
  TestProperWordSequence(sent1, {"a", "b"}, vocab);
  string lat2 = "[('a', 0), ('b', -0.1), ('c', -0.1)],[(0,1),(0,2),(1,2)]";
  Sentence sent2 = ParseAsWords(*vocab, lat2, false, "lat");
  BOOST_CHECK_EQUAL(sent2.is_lattice(), false);
  TestProperWordSequence(sent2, {"a", "b", "c"}, vocab);
}

BOOST_AUTO_TEST_CASE(TestReadLattice) {
  DictPtr vocab(CreateNewDict());
  string lat1 = "[('<s>', 0), ('a', -0), ('b',0), ('</s>', 0)],[(0,1),(0,2),(1,3) ,(2, 3)]";
  Sentence sent1 = ParseAsLattice(*vocab, lat1, "lat");
  BOOST_CHECK_EQUAL(sent1.is_lattice(), true);
  TestNumEdges(sent1, {0, 1, 1, 2});
  TestProperWordSequence(sent1, {"<s>", "a", "b", "</s>"}, vocab, false); // checking only nodes, not edges
}

BOOST_AUTO_TEST_CASE(TestReadMalformedLattice) {
  DictPtr vocab(CreateNewDict());
  BOOST_CHECK_THROW( ParseAsWords(*vocab, "a b", false, "lat"), std::runtime_error );
}

BOOST_AUTO_TEST_CASE(TestReadLatticeWithoutEdges) {
  DictPtr vocab(CreateNewDict());
  string lat1 = "[('<s>', 0), ('a', -0), ('b',0), ('</s>', 0)],[]";
  Sentence sent1 = ParseAsLattice(*vocab, lat1, "lat");
  TestProperWordSequence(sent1, {"<s>", "a", "b", "</s>"}, vocab, false);
  TestNumEdges(sent1, {0, 0, 0, 0});
  BOOST_CHECK_EQUAL(sent1.is_feature_represented(), false);
}

BOOST_AUTO_TEST_CASE(TestReadLatticeWithoutNodes) {
  DictPtr vocab(CreateNewDict());
  string lat1 = "[ ], []";
  BOOST_CHECK_THROW( ParseAsLattice(*vocab, lat1, "lat"), std::runtime_error );
  string lat2 = "[('a',0)], []";
  BOOST_CHECK_THROW( ParseAsLattice(*vocab, lat2, "lat"), std::runtime_error );
}

BOOST_AUTO_TEST_CASE(TestReadIllegalLattice) {
  DictPtr vocab(CreateNewDict());
  string lat1 = "[('<s>', 0), ('a', -0), ('b',0), ('</s>', 0)],[(1,2),(2,6)]";
  BOOST_CHECK_THROW( ParseAsLattice(*vocab, lat1, "lat"), std::runtime_error );
}

BOOST_AUTO_TEST_CASE(TestReadWordsAsLattice) {
  DictPtr vocab(CreateNewDict());
  Sentence sent1 = ParseAsLattice(*vocab, "a b", "txt");
  BOOST_CHECK_EQUAL(sent1.is_lattice(), true);
  TestProperWordSequence(sent1, {"a", "b"}, vocab);
}

BOOST_AUTO_TEST_CASE(TestReadFeatures, * utf::tolerance(0.001)) {
  Sentence sent1 = ParseAsFeatures("1 2 3 ;    .1 .2 .3;      -0 -1.1 -3e-3", "feat");
  BOOST_CHECK_EQUAL(sent1.is_lattice(), false);
  BOOST_CHECK_EQUAL(sent1.is_feature_represented(), true);
  BOOST_CHECK_EQUAL(sent1.size(), 3);
  BOOST_CHECK_EQUAL(sent1.get_feature_dim(), 3);
  BOOST_CHECK_EQUAL(sent1.get_feature(0).size(), 3);
  BOOST_CHECK_EQUAL(sent1.get_feature(1).size(), 3);
  BOOST_CHECK_EQUAL(sent1.get_feature(2).size(), 3);
  BOOST_TEST(sent1.get_feature(0)[0] == 1);
  BOOST_TEST(sent1.get_feature(0)[1] == 2);
  BOOST_TEST(sent1.get_feature(0)[2] == 3);
  BOOST_TEST(sent1.get_feature(1)[0] == .1);
  BOOST_TEST(sent1.get_feature(1)[1] == .2);
  BOOST_TEST(sent1.get_feature(1)[2] == .3);
  BOOST_TEST(sent1.get_feature(2)[0] == 0);
  BOOST_TEST(sent1.get_feature(2)[1] == -1.1);
  BOOST_TEST(sent1.get_feature(2)[2] == -3e-3);
}

BOOST_AUTO_TEST_CASE(TestReadFeaturesShort) {
  Sentence sent1 = ParseAsFeatures("1 2 3", "feat");
  BOOST_CHECK_EQUAL(sent1.is_lattice(), false);
  BOOST_CHECK_EQUAL(sent1.is_feature_represented(), true);
  BOOST_CHECK_EQUAL(sent1.get_feature_dim(), 3);
  BOOST_CHECK_EQUAL(sent1.get_feature(0).size(), 3);
  BOOST_CHECK_EQUAL(sent1.size(), 1);
}

BOOST_AUTO_TEST_CASE(TestReadFeaturesInconsistent) {
  BOOST_CHECK_THROW( ParseAsFeatures("1 2 3 ;    .1 .2 .3; 0 0", "feat"), std::runtime_error );
}

BOOST_AUTO_TEST_CASE(TestReadFeaturesMalformed) {
  BOOST_CHECK_THROW(ParseAsFeatures("1 2 3 ;    .1 .2 .3,0 0 0", "feat"), std::runtime_error );
  BOOST_CHECK_THROW(ParseAsFeatures("; 1 2 3 ;    .1 .2 .3,0 0 0", "feat"), std::runtime_error );
  BOOST_CHECK_THROW(ParseAsFeatures("a b c", "feat"), std::runtime_error );
//  BOOST_CHECK_THROW(ParseAsFeatures(" ", "feat"), std::runtime_error );
}

BOOST_AUTO_TEST_SUITE_END()
