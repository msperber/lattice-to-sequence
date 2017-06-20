#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>
#include "dynet/lattice-lstm.h"
#include "dynet/vanilla-lstm.h"

using namespace dynet;
using namespace dynet::expr;
using namespace std;


struct LatticeLstmRnnEqualityTest {
  LatticeLstmRnnEqualityTest() {
    // initialize if necessary
    if(default_device == nullptr) {
      for (auto x : {"LatticeLstmTest", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }

  }
  ~LatticeLstmRnnEqualityTest() {
    for (auto x : av) free(x);
  }

  template <class T>
  std::string print_vec(const std::vector<T> vec) {
    ostringstream oss;
    if(vec.size()) oss << vec[0];
    for(size_t i = 1; i < vec.size(); i++)
      oss << ' ' << vec[i];
    return oss.str();
  }

  std::vector<char*> av;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(lattice_lstm_rnn_equality_test, LatticeLstmRnnEqualityTest);



BOOST_AUTO_TEST_CASE( lattice_and_naive_lstm_equal_single_input ) {

  /*
   * forward pass for lattice LSTM for some toy single input
   */
  dynet::expr::Expression i_wr_1, i_h_1, i_wr_2, i_h_2, i_wr_3, i_h_3, i_wr_4, i_h_4;
  std::shared_ptr<dynet::LatticeLSTMBuilder> builder_lattice_;
  unsigned n_layers = 1;
  unsigned wordrep_size = 5;
  unsigned n_nodes = 4;
  dynet::Model model;
  dynet::ComputationGraph cg;
  builder_lattice_ = shared_ptr<dynet::LatticeLSTMBuilder>(
      new dynet::LatticeLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lattice_->new_graph(cg);
  builder_lattice_->start_new_sequence();
  i_wr_1 = input(cg, dynet::Dim({wordrep_size}), {-1,0,1,1,1});
  i_h_1 = builder_lattice_->add_input(i_wr_1);
  Tensor o_lattice = cg.forward(i_h_1);

  /*
   * do the same thing manually with some standard LSTM code
   * (derived from lstm.cc, but w/ removed input-forget gate coupling and removed peepholes)
   */
  enum { X2I, H2I, BI, X2O, H2O, BO, X2F, H2F, BF, X2C, H2C, BC };
  //i
  std::vector<Expression> vars;
  Expression i_x2i = parameter(cg, builder_lattice_->params[0][X2I]);
  Expression i_h2i = parameter(cg, builder_lattice_->params[0][H2I]);
  Expression i_bi = parameter(cg, builder_lattice_->params[0][BI]);
  //f
  Expression i_x2f = parameter(cg, builder_lattice_->params[0][X2F]);
  Expression i_h2f = parameter(cg, builder_lattice_->params[0][H2F]);
  Expression i_bf = parameter(cg, builder_lattice_->params[0][BF]);
  //o
  Expression i_x2o = parameter(cg, builder_lattice_->params[0][X2O]);
  Expression i_h2o = parameter(cg, builder_lattice_->params[0][H2O]);
  Expression i_bo = parameter(cg, builder_lattice_->params[0][BO]);
  //c
  Expression i_x2c = parameter(cg, builder_lattice_->params[0][X2C]);
  Expression i_h2c = parameter(cg, builder_lattice_->params[0][H2C]);
  Expression i_bc = parameter(cg, builder_lattice_->params[0][BC]);

  vars = {i_x2i, i_h2i, i_bi, i_x2o, i_h2o, i_bo, i_x2f, i_h2f, i_bf, i_x2c, i_h2c, i_bc};
//  param_vars.push_back(vars);
  std::vector<Expression> h0, c0;
  h0.resize(1);
  c0.resize(1);


  vector<Expression>& ht = h0;
  vector<Expression>& ct = c0;
  Expression in = i_wr_1;
  Expression i_h_tm1, i_c_tm1;
  // input
  Expression i_ait;
  i_ait = affine_transform({vars[BI], vars[X2I], in});
  Expression i_it = logistic(i_ait);
  // forget
  Expression i_aft;
  i_aft = affine_transform({vars[BF], vars[X2F], in});
  Expression i_ft = logistic(i_aft);
//  Expression i_ft = 1.f - i_it;
  // write memory cell
  Expression i_awt;
  i_awt = affine_transform({vars[BC], vars[X2C], in});
  Expression i_wt = tanh(i_awt);
  // output
  ct[0] = cmult(i_it, i_wt);

  Expression i_aot;
  i_aot = affine_transform({vars[BO], vars[X2O], in});
  Expression i_ot = logistic(i_aot);
  Expression ph_t = tanh(ct[0]);
  ht[0] = cmult(i_ot, ph_t);
  Tensor o_linear = cg.forward(ht.back());

  BOOST_CHECK_EQUAL(*o_lattice.v, *o_linear.v);


}

BOOST_AUTO_TEST_CASE( lattice_and_naive_lstm_equal_seq_input ) {

  /*
   * forward pass for lattice LSTM for seq of 4 toy inputs
   */
  dynet::expr::Expression i_wr_1, i_h_1, i_wr_2, i_h_2, i_wr_3, i_h_3, i_wr_4, i_h_4;
  std::shared_ptr<dynet::LatticeLSTMBuilder> builder_lattice_;
  unsigned n_layers = 1;
  unsigned wordrep_size = 5;
  unsigned n_nodes = 4;
  dynet::Model model;
  dynet::ComputationGraph cg;
  builder_lattice_ = shared_ptr<dynet::LatticeLSTMBuilder>(
      new dynet::LatticeLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lattice_->new_graph(cg);
  builder_lattice_->start_new_sequence();
  i_wr_1 = input(cg, dynet::Dim({wordrep_size}), {-1,0,1,1,1});
  i_wr_2 = input(cg, dynet::Dim({wordrep_size}), {-1,1,0,1,1});
  i_wr_3 = input(cg, dynet::Dim({wordrep_size}), {0,0,1,1,1});
  i_wr_4 = input(cg, dynet::Dim({wordrep_size}), {-1,1,2,1,1});
  i_h_1 = builder_lattice_->add_input(i_wr_1);
  i_h_2 = builder_lattice_->add_input(i_wr_2);
  i_h_3 = builder_lattice_->add_input(i_wr_3);
  i_h_4 = builder_lattice_->add_input(i_wr_4);
  Tensor o_lattice = cg.forward(i_h_4);

  /*
   * do the same thing manually with some standard LSTM code
   * (derived from lstm.cc, but w/ removed input-forget gate coupling and removed peepholes)
   */
  enum { X2I, H2I, BI, X2O, H2O, BO, X2F, H2F, BF, X2C, H2C, BC };
  //i
  std::vector<Expression> vars;
  Expression i_x2i = parameter(cg, builder_lattice_->params[0][X2I]);
  Expression i_h2i = parameter(cg, builder_lattice_->params[0][H2I]);
  Expression i_bi = parameter(cg, builder_lattice_->params[0][BI]);
  //f
  Expression i_x2f = parameter(cg, builder_lattice_->params[0][X2F]);
  Expression i_h2f = parameter(cg, builder_lattice_->params[0][H2F]);
  Expression i_bf = parameter(cg, builder_lattice_->params[0][BF]);
  //o
  Expression i_x2o = parameter(cg, builder_lattice_->params[0][X2O]);
  Expression i_h2o = parameter(cg, builder_lattice_->params[0][H2O]);
  Expression i_bo = parameter(cg, builder_lattice_->params[0][BO]);
  //c
  Expression i_x2c = parameter(cg, builder_lattice_->params[0][X2C]);
  Expression i_h2c = parameter(cg, builder_lattice_->params[0][H2C]);
  Expression i_bc = parameter(cg, builder_lattice_->params[0][BC]);

  vars = {i_x2i, i_h2i, i_bi, i_x2o, i_h2o, i_bo, i_x2f, i_h2f, i_bf, i_x2c, i_h2c, i_bc};
//  param_vars.push_back(vars);
  std::vector<std::vector<Expression>> h, c;
  std::vector<Expression> h0, c0;
  h0.resize(1);
  c0.resize(1);

  Expression i_h_tm1, i_c_tm1;
  vector<Expression> x = {i_wr_1,i_wr_2,i_wr_3,i_wr_4};
  Tensor o_linear;
  for(int prev=-1; prev<3; ++prev){
    h.push_back(vector<Expression>(1));
    c.push_back(vector<Expression>(1));
    vector<Expression>& ht = h.back();
    vector<Expression>& ct = c.back();
    Expression in = x[prev+1];
    bool has_prev_state = prev >= 0;
    if (prev >= 0) { // t > 0
      i_h_tm1 = h[prev][0];
      i_c_tm1 = c[prev][0];
    }
    // input
    Expression i_ait;
    if (has_prev_state)
      i_ait = affine_transform({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1});
    else
      i_ait = affine_transform({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);
    // forget
//    Expression i_ft = 1.f - i_it;
    Expression i_aft;
    if (has_prev_state)
      i_aft = affine_transform({vars[BF], vars[X2F], in, vars[H2F], i_h_tm1});
    else
      i_aft = affine_transform({vars[BF], vars[X2F], in});
    Expression i_ft = logistic(i_aft);
    // write memory cell
    Expression i_awt;
    if (has_prev_state)
      i_awt = affine_transform({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
      i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);
    // output
    if (has_prev_state) {
      Expression i_nwt = cmult(i_it, i_wt);
      Expression i_crt = cmult(i_ft, i_c_tm1);
      ct[0] = i_crt + i_nwt;
    } else {
      ct[0] = cmult(i_it, i_wt);
    }

    Expression i_aot;
    if (has_prev_state)
      i_aot = affine_transform({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1});
    else
      i_aot = affine_transform({vars[BO], vars[X2O], in});
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(ct[0]);
    ht[0] = cmult(i_ot, ph_t);
    o_linear = cg.forward(ht.back());
  }

  BOOST_CHECK_EQUAL(*o_lattice.v, *o_linear.v);


}


BOOST_AUTO_TEST_CASE( lattice_lstm_order_invariant ) {

  /*
   * forward pass for lattice LSTM for 2 toy inputs
   */
  dynet::expr::Expression i_wr_1, i_h_1, i_wr_2, i_h_2, i_wr_3, i_h_3, i_wr_4, i_h_4, i_wr_5, i_h_5;
  std::shared_ptr<dynet::LatticeLSTMBuilder> builder_lattice_;
  unsigned n_layers = 2;
  unsigned wordrep_size = 5;
  unsigned n_nodes = 4;
  dynet::Model model;
  dynet::ComputationGraph cg;
  builder_lattice_ = shared_ptr<dynet::LatticeLSTMBuilder>(
      new dynet::LatticeLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lattice_->new_graph(cg);
  builder_lattice_->start_new_sequence();
  i_wr_1 = input(cg, dynet::Dim({wordrep_size}), {-1,0,1,0,1});
  i_wr_2 = input(cg, dynet::Dim({wordrep_size}), {-1,1,0,0,1});
  i_wr_3 = input(cg, dynet::Dim({wordrep_size}), {-1,0,0,0,1});
  i_wr_4 = input(cg, dynet::Dim({wordrep_size}), {-1,0,0,0,1});
  i_wr_5 = input(cg, dynet::Dim({wordrep_size}), {0,0,0,0,0});
  i_h_1 = builder_lattice_->add_input_multichild({}, i_wr_1);
  i_h_2 = builder_lattice_->add_input_multichild({}, i_wr_2);
  i_h_3 = builder_lattice_->add_input_multichild({0}, i_wr_3);
  i_h_4 = builder_lattice_->add_input_multichild({0,1}, i_wr_4);
  i_h_5 = builder_lattice_->add_input_multichild({2,3}, i_wr_5);
  Tensor o_lattice_1 = cg.forward(i_h_5);

  builder_lattice_->start_new_sequence();
  i_h_1 = builder_lattice_->add_input_multichild({}, i_wr_2);
  i_h_2 = builder_lattice_->add_input_multichild({}, i_wr_1);
  i_h_3 = builder_lattice_->add_input_multichild({0,1}, i_wr_4);
  i_h_4 = builder_lattice_->add_input_multichild({1}, i_wr_3);
  i_h_5 = builder_lattice_->add_input_multichild({2,3}, i_wr_5);
  Tensor o_lattice_2 = cg.forward(i_h_5);

  BOOST_CHECK_EQUAL(*o_lattice_1.v, *o_lattice_2.v);


}


BOOST_AUTO_TEST_CASE( lattice_and_vanilla_lstm_equal_1 ) {

  unsigned n_layers = 1;
  unsigned wordrep_size = 5;
  unsigned n_nodes = 4;

  dynet::Model model;
  dynet::ComputationGraph cg;

  dynet::expr::Expression i_wr_1, i_wr_2, i_wr_3, i_wr_4;

  /*
   * forward pass for lattice LSTM for some toy single input
   */
  dynet::expr::Expression i_h_1, i_h_2, i_h_3, i_h_4;
  std::shared_ptr<dynet::LatticeLSTMBuilder> builder_lattice_;
  builder_lattice_ = shared_ptr<dynet::LatticeLSTMBuilder>(
      new dynet::LatticeLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lattice_->new_graph(cg);
  builder_lattice_->start_new_sequence();
  i_wr_1 = input(cg, dynet::Dim({wordrep_size}), {-1,0,1,0,0});
  i_wr_2 = input(cg, dynet::Dim({wordrep_size}), {0,0,0,0,0});
  i_wr_3 = input(cg, dynet::Dim({wordrep_size}), {-1,0,1,0,0});
  i_wr_4 = input(cg, dynet::Dim({wordrep_size}), {1,0,-1,0,0});
  i_h_1 = builder_lattice_->add_input(i_wr_1);
  i_h_2 = builder_lattice_->add_input(i_wr_2);
  i_h_3 = builder_lattice_->add_input(i_wr_3);
  i_h_4 = builder_lattice_->add_input(i_wr_4);
  builder_lattice_->start_new_sequence();
  i_h_4 = builder_lattice_->add_input(i_wr_4);
  i_h_3 = builder_lattice_->add_input(i_wr_3);
  i_h_2 = builder_lattice_->add_input(i_wr_2);
  i_h_1 = builder_lattice_->add_input(i_wr_1);
  Tensor o_lattice = cg.forward(i_h_1);


  /*
   * same for vanilla LSTM
   */
  dynet::expr::Expression i_h_1_l, i_h_2_l, i_h_3_l, i_h_4_l;
  std::shared_ptr<dynet::VanillaLSTMBuilder> builder_lstm_;
//  dynet::Model model;
//  dynet::ComputationGraph cg;
  builder_lstm_ = shared_ptr<dynet::VanillaLSTMBuilder>(
      new dynet::VanillaLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lstm_->params = builder_lattice_->params;
  builder_lstm_->new_graph(cg);
  builder_lstm_->start_new_sequence();
  i_h_1_l = builder_lstm_->add_input(i_wr_1);
  i_h_2_l = builder_lstm_->add_input(i_wr_2);
  i_h_3_l = builder_lstm_->add_input(i_wr_3);
  i_h_4_l = builder_lstm_->add_input(i_wr_4);
  builder_lstm_->start_new_sequence();
  i_h_4_l = builder_lstm_->add_input(i_wr_4);
  i_h_3_l = builder_lstm_->add_input(i_wr_3);
  i_h_2_l = builder_lstm_->add_input(i_wr_2);
  i_h_1_l = builder_lstm_->add_input(i_wr_1);
  Tensor o_linear = cg.forward(i_h_1_l);

  BOOST_CHECK_EQUAL(*o_lattice.v, *o_linear.v);

}


BOOST_AUTO_TEST_CASE( lattice_and_vanilla_lstm_equal_minibatch ) {

  unsigned n_layers = 1;
  unsigned wordrep_size = 5;
  unsigned n_nodes = 4;
  unsigned batch_size = 2;

  dynet::Model model;
  dynet::ComputationGraph cg;

  dynet::expr::Expression i_wr_1, i_wr_2;

  /*
   * forward pass for lattice LSTM for some toy single input
   */
  dynet::expr::Expression i_h_1;
  std::shared_ptr<dynet::LatticeLSTMBuilder> builder_lattice_;
  builder_lattice_ = shared_ptr<dynet::LatticeLSTMBuilder>(
      new dynet::LatticeLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lattice_->new_graph(cg);
  i_wr_1 = input(cg, dynet::Dim({wordrep_size}, batch_size), {-1,0,1,0,0, -1,0,1,0,0});
  i_wr_2 = input(cg, dynet::Dim({wordrep_size}, batch_size), {0,1,0,0,0, 0,1,0,0,0});

  builder_lattice_->start_new_sequence();
  builder_lattice_->add_input(i_wr_1);
  i_h_1 = builder_lattice_->add_input(i_wr_2);
  Tensor o_lattice = cg.forward(i_h_1);


  /*
   * same for vanilla LSTM
   */
  dynet::expr::Expression i_h_1_l;
  std::shared_ptr<dynet::VanillaLSTMBuilder> builder_lstm_;
//  dynet::Model model;
//  dynet::ComputationGraph cg;
  builder_lstm_ = shared_ptr<dynet::VanillaLSTMBuilder>(
      new dynet::VanillaLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lstm_->params = builder_lattice_->params;
  builder_lstm_->new_graph(cg);
  i_wr_1 = input(cg, dynet::Dim({wordrep_size}, batch_size), {-1,0,1,0,0, -1,0,1,0,0});
  i_wr_2 = input(cg, dynet::Dim({wordrep_size}, batch_size), {0,1,0,0,0, 0,1,0,0,0});

  builder_lstm_->start_new_sequence();
  builder_lstm_->add_input(i_wr_1);
  i_h_1_l = builder_lstm_->add_input(i_wr_2);
  Tensor o_linear = cg.forward(i_h_1_l);

  BOOST_CHECK_EQUAL(*o_lattice.v, *o_linear.v);

}


BOOST_AUTO_TEST_CASE( lattice_and_vanilla_lstm_equal_minibatch1 ) {

  unsigned n_layers = 1;
  unsigned wordrep_size = 5;
  unsigned n_nodes = 4;
  unsigned batch_size = 1;

  dynet::Model model;
  dynet::ComputationGraph cg;

  dynet::expr::Expression i_wr_1, i_wr_2;

  /*
   * forward pass for lattice LSTM for some toy single input
   */
  dynet::expr::Expression i_h_1;
  std::shared_ptr<dynet::LatticeLSTMBuilder> builder_lattice_;
  builder_lattice_ = shared_ptr<dynet::LatticeLSTMBuilder>(
      new dynet::LatticeLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lattice_->new_graph(cg);
  i_wr_1 = input(cg, dynet::Dim({wordrep_size}, batch_size), {-1,0,1,0,0, -1,0,1,0,0});
  i_wr_2 = input(cg, dynet::Dim({wordrep_size}, batch_size), {0,1,0,0,0, 0,1,0,0,0});

  builder_lattice_->start_new_sequence();
  builder_lattice_->add_input(i_wr_1);
  i_h_1 = builder_lattice_->add_input(i_wr_2);
  Tensor o_lattice = cg.forward(i_h_1);


  /*
   * same for data without batch size
   */
  dynet::expr::Expression i_wr_1_l, i_wr_2_l;
  dynet::expr::Expression i_h_1_l;
  std::shared_ptr<dynet::VanillaLSTMBuilder> builder_lstm_;
//  dynet::Model model;
//  dynet::ComputationGraph cg;
//  builder_lstm_ = shared_ptr<dynet::VanillaLSTMBuilder>(
//      new dynet::VanillaLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
//  builder_lstm_->params = builder_lattice_->params;
  builder_lattice_->new_graph(cg);
  i_wr_1_l = input(cg, dynet::Dim({wordrep_size}), {-1,0,1,0,0, -1,0,1,0,0});
  i_wr_2_l = input(cg, dynet::Dim({wordrep_size}), {0,1,0,0,0, 0,1,0,0,0});

  builder_lattice_->start_new_sequence();
  builder_lattice_->add_input(i_wr_1_l);
  i_h_1_l = builder_lattice_->add_input(i_wr_2_l);
  Tensor o_lattice1 = cg.forward(i_h_1_l);

  BOOST_CHECK_EQUAL(*o_lattice.v, *o_lattice1.v);

}

BOOST_AUTO_TEST_CASE( lattice_lstm_scores_order_invariant ) {

  /*
   * forward pass for lattice LSTM for 2 toy inputs
   */
  dynet::expr::Expression i_wr_1, i_h_1, i_wr_2, i_h_2, i_wr_3, i_h_3, i_wr_4, i_h_4, i_wr_5, i_h_5;
  dynet::expr::Expression i_s_01, i_s_02, i_s_05, i_s_1;
  std::shared_ptr<dynet::LatticeLSTMBuilder> builder_lattice_;
  unsigned n_layers = 2;
  unsigned wordrep_size = 5;
  unsigned n_nodes = 4;
  dynet::Model model;
  dynet::ComputationGraph cg;
  builder_lattice_ = shared_ptr<dynet::LatticeLSTMBuilder>(
      new dynet::LatticeLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lattice_->new_graph(cg);
  builder_lattice_->start_new_sequence();
  i_wr_1 = input(cg, dynet::Dim({wordrep_size}), {-1,0,1,0,1});
  i_wr_2 = input(cg, dynet::Dim({wordrep_size}), {-1,1,0,0,1});
  i_wr_3 = input(cg, dynet::Dim({wordrep_size}), {-1,0,0,0,1});
  i_wr_4 = input(cg, dynet::Dim({wordrep_size}), {-1,0,0,0,1});
  i_wr_5 = input(cg, dynet::Dim({wordrep_size}), {0,0,0,0,0});
  i_s_01 = input(cg, dynet::Dim({1}), {0.1});
  i_s_02 = input(cg, dynet::Dim({1}), {0.2});
  i_s_05 = input(cg, dynet::Dim({1}), {0.5});
  i_s_1 = input(cg, dynet::Dim({1}), {1});
  i_h_1 = builder_lattice_->add_input_multichild({}, i_wr_1, {});
  i_h_2 = builder_lattice_->add_input_multichild({}, i_wr_2, {});
  i_h_3 = builder_lattice_->add_input_multichild({0}, i_wr_3, {i_s_1});
  i_h_4 = builder_lattice_->add_input_multichild({0,1}, i_wr_4, {i_s_05, i_s_05});
  i_h_5 = builder_lattice_->add_input_multichild({2,3}, i_wr_5, {i_s_05, i_s_05});
  Tensor o_lattice_1 = cg.forward(i_h_5);

  builder_lattice_->start_new_sequence();
  i_h_1 = builder_lattice_->add_input_multichild({}, i_wr_2, {});
  i_h_2 = builder_lattice_->add_input_multichild({}, i_wr_1, {});
  i_h_3 = builder_lattice_->add_input_multichild({0,1}, i_wr_4, {i_s_05, i_s_05});
  i_h_4 = builder_lattice_->add_input_multichild({1}, i_wr_3, {i_s_1});
  i_h_5 = builder_lattice_->add_input_multichild({2,3}, i_wr_5, {i_s_05, i_s_05});
  Tensor o_lattice_2 = cg.forward(i_h_5);

  BOOST_CHECK_EQUAL(*o_lattice_1.v, *o_lattice_2.v);
}


BOOST_AUTO_TEST_CASE( lattice_lstm_scores_scale_invariant ) {

  /*
   * forward pass for lattice LSTM for 2 toy inputs
   */
  dynet::expr::Expression i_wr_1, i_h_1, i_wr_2, i_h_2, i_wr_3, i_h_3, i_wr_4, i_h_4, i_wr_5, i_h_5;
  dynet::expr::Expression i_s_01, i_s_02, i_s_05, i_s_1;
  std::shared_ptr<dynet::LatticeLSTMBuilder> builder_lattice_;
  unsigned n_layers = 1;
  unsigned wordrep_size = 5;
  unsigned n_nodes = 4;
  dynet::Model model;
  dynet::ComputationGraph cg;
  builder_lattice_ = shared_ptr<dynet::LatticeLSTMBuilder>(
      new dynet::LatticeLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
  builder_lattice_->new_graph(cg);
  builder_lattice_->start_new_sequence();
  i_wr_1 = input(cg, dynet::Dim({wordrep_size}), {-1,0,1,0,1});
  i_wr_2 = input(cg, dynet::Dim({wordrep_size}), {-1,1,0,0,1});
  i_wr_3 = input(cg, dynet::Dim({wordrep_size}), {-1,0,0,0,1});
  i_wr_4 = input(cg, dynet::Dim({wordrep_size}), {-1,0,0,0,1});
  i_wr_5 = input(cg, dynet::Dim({wordrep_size}), {0,0,0,0,0});
  i_s_01  = input(cg, dynet::Dim({1}), {0.1});
  i_s_02 = input(cg, dynet::Dim({1}), {0.2});
  i_s_05 = input(cg, dynet::Dim({1}), {0.5});
  i_s_1 = input(cg, dynet::Dim({1}), {1});
  i_h_1 = builder_lattice_->add_input_multichild({}, i_wr_1, {});
  i_h_2 = builder_lattice_->add_input_multichild({}, i_wr_2, {});
  i_h_3 = builder_lattice_->add_input_multichild({0}, i_wr_3, {i_s_01});
  i_h_4 = builder_lattice_->add_input_multichild({0,1}, i_wr_4, {i_s_02, i_s_02});
  i_h_5 = builder_lattice_->add_input_multichild({2,3}, i_wr_5, {i_s_05, i_s_05});
  Tensor o_lattice_1 = cg.forward(i_h_5);

  builder_lattice_->start_new_sequence();
  i_h_1 = builder_lattice_->add_input_multichild({}, i_wr_1, {});
  i_h_2 = builder_lattice_->add_input_multichild({}, i_wr_2, {});
  i_h_3 = builder_lattice_->add_input_multichild({0}, i_wr_3, {i_s_1});
  i_h_4 = builder_lattice_->add_input_multichild({0,1}, i_wr_4, {i_s_05, i_s_05});
  i_h_5 = builder_lattice_->add_input_multichild({2,3}, i_wr_5, {i_s_02, i_s_02});
  Tensor o_lattice_2 = cg.forward(i_h_5);

  BOOST_CHECK_EQUAL(*o_lattice_1.v, *o_lattice_2.v);
}




BOOST_AUTO_TEST_SUITE_END()


