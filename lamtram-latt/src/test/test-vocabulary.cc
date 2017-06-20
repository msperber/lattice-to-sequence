#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <lamtram/macros.h>
#include <lamtram/sentence.h>
#include <lamtram/dict-utils.h>
#include <dynet/dict.h>
#include <sstream>

using namespace std;
using namespace lamtram;

// ****** The tests *******
BOOST_AUTO_TEST_SUITE(vocabulary)

BOOST_AUTO_TEST_CASE(TestParseWords) {
    // Create the words. <s> is 0 and <unk> is 1.
    DictPtr vocab(CreateNewDict());
    string in = "  a b  c a c <s> b <unk> ";
    Sentence exp(8), act;
    exp[0] = 2;
    exp[1] = 3;
    exp[2] = 4;
    exp[3] = 2;
    exp[4] = 4;
    exp[5] = 0;
    exp[6] = 3;
    exp[7] = 1;
    act = ParseAsWords(*vocab, in, false);
    // there was some weird (unpredictable) behavior, where the check failed reporting ridiculous vector sizes, even though printing them out looked exactly the same.
    //BOOST_CHECK_EQUAL_COLLECTIONS(exp.get_word_ids().begin(), exp.get_word_ids().end(), act.get_word_ids().begin(), act.get_word_ids().end());
    // the less pretty version seems to be ok:
    BOOST_CHECK_EQUAL(exp.size(), act.size());
    for(int i=0; i<act.size(); i++) BOOST_CHECK_EQUAL(exp[i], act[i]);
}

BOOST_AUTO_TEST_CASE(TestPrintWords) {
    dynet::Dict vocab;
    string in = "  a b  c a c <s> b  ";
    string exp = "a b c a c <s> b";
    string act = PrintWords(vocab, ParseAsWords(vocab, in, false));
    BOOST_CHECK_EQUAL(exp, act);
}

BOOST_AUTO_TEST_CASE(TestReadWrite) {
    DictPtr vocab_exp(CreateNewDict());
    string in = "  a b  c a c <s> b  ";
    ParseAsWords(*vocab_exp, in, false);
    stringstream out;
    WriteDict(*vocab_exp, out);
    DictPtr vocab_act(ReadDict(out));
    BOOST_CHECK_EQUAL_COLLECTIONS(
        vocab_exp->get_words().begin(), vocab_exp->get_words().end(),
        vocab_act->get_words().begin(), vocab_act->get_words().end());
}


BOOST_AUTO_TEST_SUITE_END()
