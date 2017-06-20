#include "dynet/lattice-lstm.h"

#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "dynet/nodes.h"
#include "dynet/io-macros.h"


using namespace std;
using namespace dynet::expr;

namespace dynet {

enum { X2I, H2I, BI, X2O, H2O, BO, X2F, H2F, BF, X2C, H2C, BC, TC, TF };

LatticeLSTMBuilder::LatticeLSTMBuilder(unsigned layers,
                                       unsigned input_dim,
                                       unsigned hidden_dim,
                                       Model* model,
				       bool update_chs_latt_temp, dynet::real chs_latt_temp,
				       bool update_frg_latt_temp, dynet::real frg_latt_temp)
    : layers(layers), update_chs_latt_temp_(update_chs_latt_temp), update_frg_latt_temp_(update_frg_latt_temp)  {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameter p_x2i = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bi = model->add_parameters({hidden_dim});

    // o
    Parameter p_x2o = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bo = model->add_parameters({hidden_dim});

    // f
    Parameter p_x2f = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2f = model->add_parameters({hidden_dim, hidden_dim});
    ParameterInitConst pi_const1(1.f);
    Parameter p_bf = model->add_parameters({hidden_dim}, pi_const1); // initialize forget gates to 1 as recommended by http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf

    // c
    Parameter p_x2c = model->add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2c = model->add_parameters({hidden_dim, hidden_dim});
    Parameter p_bc = model->add_parameters({hidden_dim});
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    // temperature
    ParameterInitConst pi_const_tc(chs_latt_temp);
    Parameter p_tc = model->add_parameters({layer_input_dim}, pi_const1);
    ParameterInitConst pi_const_tf(frg_latt_temp);
    Parameter p_tf = model->add_parameters({hidden_dim}, pi_const1);

    vector<Parameter> ps = {p_x2i, p_h2i, p_bi, p_x2o, p_h2o, p_bo, p_x2f, p_h2f, p_bf, p_x2c, p_h2c, p_bc, p_tc, p_tf};
    params.push_back(ps);
  }  // layers
  dropout_rate = 0.f;
}

void LatticeLSTMBuilder::new_graph_impl(ComputationGraph& cg){
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];

    //i
    Expression i_x2i = parameter(cg,p[X2I]);
    Expression i_h2i = parameter(cg,p[H2I]);
    Expression i_bi = parameter(cg,p[BI]);
    //o
    Expression i_x2o = parameter(cg,p[X2O]);
    Expression i_h2o = parameter(cg,p[H2O]);
    Expression i_bo = parameter(cg,p[BO]);
    //f
    Expression i_x2f = parameter(cg,p[X2F]);
    Expression i_h2f = parameter(cg,p[H2F]);
    Expression i_bf = parameter(cg,p[BF]);
    //c
    Expression i_x2c = parameter(cg,p[X2C]);
    Expression i_h2c = parameter(cg,p[H2C]);
    Expression i_bc = parameter(cg,p[BC]);
    //temp
    Expression i_tc = parameter(cg,p[TC]);
    if(!update_chs_latt_temp_){
      i_tc = nobackprop(i_tc);
    }
    Expression i_tf = parameter(cg,p[TF]);
    if(!update_frg_latt_temp_){
      i_tf = nobackprop(i_tf);
    }

    vector<Expression> vars = {i_x2i, i_h2i, i_bi, i_x2o, i_h2o, i_bo, i_x2f, i_h2f, i_bf, i_x2c, i_h2c, i_bc, i_tc, i_tf};
    param_vars.push_back(vars);
  }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void LatticeLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }
}

Expression LatticeLSTMBuilder::add_input(const Expression& x) {
  if(h.size()==0){
    return add_input_multichild({}, x);
  } else {
    return add_input_multichild({(int)(h.size()-1)}, x);
  }
}

Expression LatticeLSTMBuilder::add_input_multichild(const std::vector<int> prevs, const Expression& x, const std::vector<Expression> scores) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (prevs.size() > 0 || has_initial_state);
    vector<Expression> i_h_tm1_k;
    vector<Expression> i_c_tm1_k;
    vector<Expression> child_weights_s;
    vector<Expression> child_weights_b;
    if (prevs.size() == 0) {
      if (has_initial_state) {
        // initial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      assert(prevs.size()==0 || prevs[0] < h.size());

      // normalize weights
      Expression weight_sum_s;
      Expression weight_sum_b;
      for (unsigned k=0; k<prevs.size(); ++k){
        Expression weight_s;
        Expression weight_b;
	if(scores.size()>0){
	  weight_s = exp(vars[TC] * log(scores[k]));
	  weight_b = (vars[TF]) * log(scores[k]);
	  if(k==0){
	    weight_sum_s = weight_s;
	    weight_sum_b = exp(weight_b);
	  } else {
	    weight_sum_s = weight_sum_s + weight_s;
	    weight_sum_b = weight_sum_b + exp(weight_b);
	  }
	  child_weights_s.push_back(weight_s);
	  child_weights_b.push_back(weight_b);
	}
      }
      if(scores.size()>0){
	for (unsigned k=0; k<prevs.size(); ++k){
	  child_weights_s[k] = cdiv(child_weights_s[k], weight_sum_s);
	  child_weights_b[k] = child_weights_b[k] - log(weight_sum_b);
	}
	for (unsigned k=0; k<prevs.size(); ++k){
	  i_h_tm1_k.push_back(cmult(child_weights_s[k], h[prevs[k]][i]));
	  i_c_tm1_k.push_back(cmult(child_weights_s[k], c[prevs[k]][i]));
	  if(k==0) i_h_tm1 = i_h_tm1_k[k];
	  else i_h_tm1 = i_h_tm1 + i_h_tm1_k[k];
	}
      } else {
	for (unsigned k=0; k<prevs.size(); ++k){
	  i_h_tm1_k.push_back(h[prevs[k]][i]);
	  i_c_tm1_k.push_back(c[prevs[k]][i]);
	  if(k==0) i_h_tm1 = i_h_tm1_k[k];
	  else i_h_tm1 = i_h_tm1 + i_h_tm1_k[k];
	}
      }
    }



    // apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
    if (dropout_rate) in = dropout(in, dropout_rate);

    // input
    Expression i_ait;
    if (has_prev_state){
      i_ait = affine_transform({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1});
    } else {
      i_ait = affine_transform({vars[BI], vars[X2I], in});
    }
    Expression i_it = logistic(i_ait);
    // forget
    std::vector<Expression> i_ft_k;
    if (has_prev_state){
      for (unsigned k=0; k<prevs.size(); ++k){
        if(scores.size()>0){
          i_ft_k.push_back(logistic(affine_transform({vars[BF] + child_weights_b[k], vars[X2F], in, vars[H2F], i_h_tm1_k[k]})));
        } else {
          i_ft_k.push_back(logistic(affine_transform({vars[BF], vars[X2F], in, vars[H2F], i_h_tm1_k[k]})));
        }
      }
    } else {
	i_ft_k.push_back(logistic(affine_transform({vars[BF], vars[X2F], in})));
    }
    // write memory cell
    Expression i_awt;
    if (has_prev_state)
      i_awt = affine_transform({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
      i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);
    // output
    if (has_prev_state) {
      Expression i_nwt = cmult(i_it,i_wt);
      Expression i_crt_k;
      Expression i_crt;
      for(unsigned k=0; k<prevs.size(); ++k){
	i_crt_k = cmult(i_ft_k[k],i_c_tm1_k[k]);
	if(k==0) i_crt = i_crt_k;
	else i_crt = i_crt + i_crt_k;
      }
      ct[i] = i_crt + i_nwt;
    } else {
      ct[i] = cmult(i_it,i_wt);
    }

    Expression i_aot;
    if (has_prev_state) {
      i_aot = affine_transform({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1});
    } else {
      i_aot = affine_transform({vars[BO], vars[X2O], in});
    }
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cmult(i_ot,ph_t);
  }
  if (dropout_rate) return dropout(ht.back(), dropout_rate);
    else return ht.back();
}

void LatticeLSTMBuilder::copy(const RNNBuilder & rnn) {
  const LatticeLSTMBuilder & rnn_lstm = (const LatticeLSTMBuilder&)rnn;
  assert(params.size() == rnn_lstm.params.size());
  for(size_t i = 0; i < params.size(); ++i)
      for(size_t j = 0; j < params[i].size(); ++j)
        params[i][j] = rnn_lstm.params[i][j];
}

void LatticeLSTMBuilder::save_parameters_pretraining(const string& fname) const {
  cerr << "Writing Lattice LSTM parameters to " << fname << endl;
  ofstream of(fname);
  assert(of);
  boost::archive::binary_oarchive oa(of);
  std::string id = "LatticeLSTMBuilder:params";
  oa << id;
  oa << layers;
  for (unsigned i = 0; i < layers; ++i) {
    for (auto p : params[i]) {
      oa << p.get()->values;
    }
  }
}

void LatticeLSTMBuilder::load_parameters_pretraining(const string& fname) {
  cerr << "Loading Lattice LSTM parameters from " << fname << endl;
  ifstream of(fname);
  assert(of);
  boost::archive::binary_iarchive ia(of);
  std::string id;
  ia >> id;
  if (id != "LatticeLSTMBuilder:params") {
    cerr << "Bad id read\n";
    abort();
  }
  unsigned l = 0;
  ia >> l;
  if (l != layers) {
    cerr << "Bad number of layers\n";
    abort();
  }
  // TODO check other dimensions
  for (unsigned i = 0; i < layers; ++i) {
    for (auto p : params[i]) {
      ia >> p.get()->values;
    }
  }
}

template<class Archive>
void LatticeLSTMBuilder::serialize(Archive& ar, const unsigned int) {
  ar & boost::serialization::base_object<RNNBuilder>(*this);
  ar & params;
  ar & layers;
  ar & dropout_rate;
}
DYNET_SERIALIZE_IMPL(LatticeLSTMBuilder);


} // namespace dynet
