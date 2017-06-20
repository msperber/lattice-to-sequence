#include <lamtram/linear-encoder.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <dynet/model.h>
#include <dynet/nodes.h>
#include <dynet/rnn.h>
#include <boost/range/irange.hpp>
#include <boost/lexical_cast.hpp>
#include <ctime>
#include <fstream>
#include <dynet/pyramid-lstm.h>

using namespace std;
using namespace lamtram;

LinearEncoder::LinearEncoder(int vocab_size, int wordrep_size,
           const BuilderSpec & hidden_spec, int unk_id,
           dynet::Model & model, string enc_type,
	   bool update_chs_latt_temp, dynet::real chs_latt_temp,
	   bool update_frg_latt_temp, dynet::real frg_latt_temp) :
      vocab_size_(vocab_size), wordrep_size_(wordrep_size), unk_id_(unk_id), hidden_spec_(hidden_spec),
      enc_type_(enc_type), reverse_(false),
      chs_latt_temp_(chs_latt_temp), frg_latt_temp_(frg_latt_temp),
      update_chs_latt_temp_(update_chs_latt_temp), update_frg_latt_temp_(update_frg_latt_temp){
  if(!(enc_type=="feat" || enc_type=="lat" || enc_type=="seq")){
      THROW_ERROR("Expecting enc_type feat|lat|seq, but got something different:" << endl << enc_type);
  }
  // Hidden layers
  if(enc_type == "lat"){
    builder_lattice_ = shared_ptr<dynet::LatticeLSTMBuilder>(
	new dynet::LatticeLSTMBuilder(hidden_spec_.layers,
				      wordrep_size,
				      hidden_spec_.nodes,
				      &model,
				      update_chs_latt_temp, chs_latt_temp,
				      update_frg_latt_temp, frg_latt_temp));
  } else {
    builder_chain_ = BuilderFactory::CreateBuilder(hidden_spec_, wordrep_size, model);
  }
  // Word representations
  if(enc_type!="feat"){
    p_wr_W_ = model.add_lookup_parameters(vocab_size, {(unsigned int)wordrep_size});
  }
}


dynet::expr::Expression LinearEncoder::word_repr(dynet::ComputationGraph & cg, const Sentence & sent, int pos){
  // performs lookup() for word-id based sentences, or input() for for feature-vector based sentences
  dynet::expr::Expression i_wr_t;
  if(sent.is_feature_represented()){
    if(pos<0 || pos>=sent.size()){
      i_wr_t = input(cg, dynet::Dim({sent.get_feature_dim()}), sent.get_zero_feature());
    } else {
      i_wr_t = input(cg, dynet::Dim({sent.get_feature_dim()}), sent.get_feature(pos));
    }
  } else {
    i_wr_t = lookup(cg, p_wr_W_, pos>=0 && pos< sent.size()?sent[pos]:0);
  }
  return i_wr_t;
}

dynet::expr::Expression LinearEncoder::word_repr(dynet::ComputationGraph & cg, const vector<Sentence> & sent, int pos){
  // minibatch version
  dynet::expr::Expression i_wr_t;
  if(sent.size() > 0 && sent[0].is_feature_represented()){
    const unsigned feat_dim = sent[0].get_feature_dim();
    vector<dynet::real> words = vector<dynet::real>(sent.size() * feat_dim);
    for(size_t i = 0; i < sent.size(); i++){
      // TODO: inner loop -> memcpy ?
      for(size_t j = 0; j < feat_dim; j++){
	const unsigned array_index = j*sent.size()+i;
 	words[array_index] = (pos>=0 && pos < sent[i].size() ? sent[i].get_feature(pos)[j] : 0);
      }
    }
    i_wr_t = input(cg, dynet::Dim({feat_dim},(unsigned)sent.size()), words);
  } else {
    vector<unsigned> words(sent.size());
    for(size_t i = 0; i < sent.size(); i++){
      words[i] = (pos >= 0 && pos < sent[i].size() ? sent[i][pos] : 0);
    }
    i_wr_t = lookup(cg, p_wr_W_, words);
  }
  return i_wr_t;
}

size_t LinearEncoder::reset_word_states(const Sentence & sent, bool add){
  size_t input_len = sent.size() + (add?1:0);
  size_t output_len = input_len;
  // output size == input size, except for the pyramid LSTM:
  if(hidden_spec_.type == "pylstm"){
    output_len = std::static_pointer_cast<dynet::PyramidLSTMBuilder>(builder_chain_)->num_outputs_for_inputs(input_len);
  }
  if(output_len==0) output_len=1;
  word_states_.resize(output_len);
  word_states_pos = 0;
  return input_len;
}

size_t LinearEncoder::reset_word_states(const vector<Sentence> & sent, bool add){
  size_t input_len = sent[0].size();
  for(size_t i = 1; i < sent.size(); i++) input_len = max(input_len, sent[i].size());
  // Create the word states
  if(add) {
      input_len++;
  }
  size_t output_len = input_len;
  if(hidden_spec_.type == "pylstm"){
      output_len = std::static_pointer_cast<dynet::PyramidLSTMBuilder>(builder_chain_)->num_outputs_for_inputs(input_len);
  }
  if(output_len==0) output_len=1;
  word_states_.resize(output_len);
  word_states_pos = 0;
  return input_len;
}

void LinearEncoder::add_word_state_or_skip(dynet::expr::Expression & i_h_t, int pos, bool reverse){
  // add to word states, except if using pyramid LSTM and the current input doesn't reach the top layer.
  if(hidden_spec_.type != "pylstm" ||
	  (hidden_spec_.type == "pylstm" && std::static_pointer_cast<dynet::PyramidLSTMBuilder>(builder_chain_)->pyramid_height(pos) == hidden_spec_.layers)){
    if(!reverse){
      word_states_[word_states_pos] = i_h_t;
    } else {
      word_states_[word_states_.size()-word_states_pos-1] = i_h_t;
    }
    word_states_pos++;
  }
}

dynet::expr::Expression LinearEncoder::BuildSentGraph(const Sentence & sent, bool add, bool train, dynet::ComputationGraph & cg) {
  if(enc_type_ == "lat"){
      return BuildSentGraphLattice(sent, add, train, cg);
  } else {
      return BuildSentGraphWords(sent, add, train, cg);
  }
}

dynet::expr::Expression LinearEncoder::BuildSentGraph(const vector<Sentence> & sent, bool add, bool train, dynet::ComputationGraph & cg) {
  // TODO: remove [0]s
  if(enc_type_ == "lat"){
    if(sent.size()>1){
      if(sent[0].is_flat()){
        return BuildSentGraphLattice(sent, add, train, cg);
      } else {
        THROW_ERROR("Lattice-LSTM currently does not support multibatching for actual lattices, please set --minibatch_size 1 or input sequential data");
      }
    } else {
      return BuildSentGraphLattice(sent, add, train, cg);
    }
  } else {
    return BuildSentGraphWords(sent, add, train, cg);
  }
}

dynet::expr::Expression LinearEncoder::BuildSentGraphWords(const Sentence & sent, bool add, bool train, dynet::ComputationGraph & cg) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed computation graph don't match.");
  size_t input_len = reset_word_states(sent, add);
  size_t output_len = word_states_.size();
  builder_chain_->start_new_sequence();
  dynet::expr::Expression i_wr_t, i_h_t;
  if(!reverse_) {
    for(int t = 0; word_states_pos < output_len; t++) {
      i_wr_t = word_repr(cg, sent, t);
      i_h_t = builder_chain_->add_input(i_wr_t);
      add_word_state_or_skip(i_h_t, t, false);
    }
  } else {
    for(int t = input_len-1; word_states_pos < output_len; t--) {
      i_wr_t = word_repr(cg, sent, t);
      i_h_t = builder_chain_->add_input(i_wr_t);
      add_word_state_or_skip(i_h_t, t, true);
    }
  }
  return i_h_t;
}

dynet::expr::Expression LinearEncoder::BuildSentGraphWords(const vector<Sentence> & sent, bool add, bool train, dynet::ComputationGraph & cg) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed computation graph don't match.");
  assert(sent.size());
  size_t input_len = reset_word_states(sent, add);
  size_t output_len = word_states_.size();
  builder_chain_->start_new_sequence();
  dynet::expr::Expression i_wr_t, i_h_t;
  if(!reverse_) {
    for(int t = 0; word_states_pos < output_len; t++) {
      i_wr_t = word_repr(cg, sent, t);
      i_h_t = builder_chain_->add_input(i_wr_t);
      add_word_state_or_skip(i_h_t, t, false);
    }
  } else {
    for(int t = input_len-1; word_states_pos < output_len; t--) {
      i_wr_t = word_repr(cg, sent, t);
      i_h_t = builder_chain_->add_input(i_wr_t);
      add_word_state_or_skip(i_h_t, t, true);
    }
  }
  return i_h_t;
}

dynet::expr::Expression LinearEncoder::BuildSentGraphLattice(const Sentence & sent, bool add, bool train, dynet::ComputationGraph & cg) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed computation graph don't match.");
  size_t input_len = reset_word_states(sent, add);
  size_t output_len = word_states_.size();
  builder_lattice_->start_new_sequence();
  dynet::expr::Expression i_wr_t, i_h_t;
  if(!reverse_) {
    for(int t = 0; word_states_pos < output_len; t++) {
      i_wr_t = word_repr(cg, sent, t);
      vector<int> predecessors;
      if(t<sent.size() && t>=0) predecessors = sent.get_predecessors(t);
      else if(t==sent.size()) predecessors = {t-1};
      vector<float> scores = sent.get_multiple_scores_as_prob(predecessors, 2);
      vector<Expression> scores_expr;
      for(int i=0; i<scores.size(); i++) scores_expr.push_back(input(cg, dynet::Dim({1},1), {scores[i]}));
      i_h_t = builder_lattice_->add_input_multichild(predecessors, i_wr_t, scores_expr);
      add_word_state_or_skip(i_h_t, t, false);
    }
  } else {
    for(int t = input_len-1; word_states_pos < output_len; t--) {
      i_wr_t = word_repr(cg, sent, t);
      vector<int> successors;
      if(t<sent.size()-1 && t>=0) successors = sent.get_successors(t);
      else if(t==sent.size()-1) successors = {t+1};
      vector<float> scores = sent.get_multiple_scores_as_prob(successors, 0);
      for(int i=0; i<successors.size(); i++) successors[i] = sent.size() - successors[i];
      vector<Expression> scores_expr;
      for(int i=0; i<scores.size(); i++) scores_expr.push_back(input(cg, dynet::Dim({1},1), {scores[i]}));
      i_h_t = builder_lattice_->add_input_multichild(successors, i_wr_t, scores_expr);
      add_word_state_or_skip(i_h_t, t, true);
    }
  }
  if(GlobalVars::verbose >= 3) {
    cerr << "Childsum Temperature: " << builder_lattice_->params[0][12].values()->vec() << "\n";
    cerr << "Forget-gate Temperature: " << builder_lattice_->params[0][13].values()->vec() << "\n";
  }
  return i_h_t;
}



dynet::expr::Expression LinearEncoder::BuildSentGraphLattice(const vector<Sentence> & sent, bool add, bool train, dynet::ComputationGraph & cg) {
  if(&cg != curr_graph_)
    THROW_ERROR("Initialized computation graph and passed computation graph don't match.");
  assert(sent.size());
  size_t input_len = reset_word_states(sent, add);
  size_t output_len = word_states_.size();
  builder_lattice_->start_new_sequence();
  dynet::expr::Expression i_wr_t, i_h_t;
  if(!reverse_) {
    for(int t = 0; word_states_pos < output_len; t++) {
      i_wr_t = word_repr(cg, sent, t);
      if(sent.size()==1){
	vector<int> predecessors;
	if(t<sent[0].size() && t>=0) predecessors = sent[0].get_predecessors(t);
	else if(t==sent[0].size()) predecessors = {(int)sent[0].size()-1};
	vector<float> scores = sent[0].get_multiple_scores_as_prob(predecessors, 2);
        vector<Expression> scores_expr;
        for(int i=0; i<scores.size(); i++) scores_expr.push_back(input(cg, dynet::Dim({1},(unsigned)sent.size()), {scores[i]}));
	i_h_t = builder_lattice_->add_input_multichild(predecessors, i_wr_t, scores_expr);
      } else {
        i_h_t = builder_lattice_->add_input(i_wr_t);
      }
      add_word_state_or_skip(i_h_t, t, false);
    }
  } else {
    for(int t = input_len-1; word_states_pos < output_len; t--) {
      i_wr_t = word_repr(cg, sent, t);
      if(sent.size()==1){
	vector<int> successors;
	if(t<sent[0].size()-1 && t>=0){
          successors = sent[0].get_successors(t);
	} else if(t==sent[0].size()-1) {
	  successors = {(int)sent[0].size()};
	}
        vector<float> scores = sent[0].get_multiple_scores_as_prob(successors, 0);
	for(int i=0; i<successors.size(); i++) successors[i] = sent[0].size() - successors[i];
        vector<Expression> scores_expr;
        for(int i=0; i<scores.size(); i++) scores_expr.push_back(input(cg, dynet::Dim({1},(unsigned)sent.size()), {scores[i]}));
	i_h_t = builder_lattice_->add_input_multichild(successors, i_wr_t, scores_expr);
      } else {
	i_h_t = builder_lattice_->add_input(i_wr_t);
      }
      add_word_state_or_skip(i_h_t, t, true);
    }
  }
  if(GlobalVars::verbose >= 3) {
    cerr << "Childsum Temperature: " << builder_lattice_->params[0][12].values()->vec() << "\n";
    cerr << "Forget-gate Temperature: " << builder_lattice_->params[0][13].values()->vec() << "\n";
  }
  return i_h_t;
}



void LinearEncoder::NewGraph(dynet::ComputationGraph & cg) {
  if(builder_chain_ != NULL)
    builder_chain_->new_graph(cg);
  else
    builder_lattice_->new_graph(cg);
  curr_graph_ = &cg;
}

// void LinearEncoder::CopyParameters(const LinearEncoder & enc) {
//   assert(vocab_size_ == enc.vocab_size_);
//   assert(wordrep_size_ == enc.wordrep_size_);
//   assert(reverse_ == enc.reverse_);
//   p_wr_W_.copy(enc.p_wr_W_);
//   builder_.copy(enc.builder_);
// }

LinearEncoder* LinearEncoder::Read(std::istream & in, dynet::Model & model) {
  int vocab_size, wordrep_size, unk_id;
  string version_id, hidden_spec, line, reverse;
  string enc_type; // seq | lat | feat
  bool update_chs_latt_temp, update_frg_latt_temp;
  string chs_latt_temp, frg_latt_temp;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting Neural LM");
  istringstream iss(line);
  iss >> version_id >> vocab_size >> wordrep_size >> hidden_spec >> unk_id >> reverse >> enc_type >> update_chs_latt_temp >> chs_latt_temp >> update_frg_latt_temp >> frg_latt_temp;
  if(version_id != "linenc_004")
    THROW_ERROR("Expecting a Neural LM of version linenc_004, but got something different:" << endl << line);
  LinearEncoder * ret = new LinearEncoder(vocab_size, wordrep_size, BuilderSpec(hidden_spec), unk_id, model, enc_type,
					  update_chs_latt_temp, boost::lexical_cast<dynet::real>(chs_latt_temp),
					  update_frg_latt_temp, boost::lexical_cast<dynet::real>(frg_latt_temp));
  if(reverse == "rev") ret->SetReverse(true);
  return ret;
}
void LinearEncoder::Write(std::ostream & out) {
  out << "linenc_004 " << vocab_size_ << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << " " << (reverse_?"rev":"for") << " " << enc_type_ << " " << update_chs_latt_temp_ << " " << chs_latt_temp_ << " " << update_frg_latt_temp_ << " " << frg_latt_temp_ << endl;
}

vector<dynet::expr::Expression> LinearEncoder::GetFinalHiddenLayers() const {
  if(builder_chain_ != NULL)
    return builder_chain_->final_h();
  else
    return builder_lattice_->final_h();
}

void LinearEncoder::SetDropout(float dropout) {
  if(builder_chain_ != NULL)
    builder_chain_->set_dropout(dropout);
  else
    builder_lattice_->set_dropout(dropout);
}
