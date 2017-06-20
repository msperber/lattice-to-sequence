#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>
#include <dynet/lstm.h>
#include <dynet/pyramid-lstm.h>

using namespace dynet;
using namespace dynet::expr;
using namespace std;


struct PyramidLstmTest {
  PyramidLstmTest() {
    // initialize if necessary
    if(default_device == nullptr) {
      for (auto x : {"PyramidLstmTest", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }
    wordrep_size = 3;
    n_nodes = 5;
    desired = 0;
    out_size = 6;


  }
  ~PyramidLstmTest() {
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

  unsigned n_layers;
  unsigned wordrep_size;
  unsigned n_nodes;
  unsigned desired;
  unsigned out_size;

  dynet::ComputationGraph cg;
  dynet::Model model;
  Parameter p_R;
  Parameter p_bias;
  std::shared_ptr<dynet::LSTMBuilder> builder_rnn;
  dynet::expr::Expression i_wr_1, i_h_1, i_wr_2, i_h_2, i_wr_3, i_h_3, i_wr_4, i_h_4, i_wr_5, i_h_5;
  Expression R;
  Expression bias;
  Expression u_t;
  Tensor e_rnn;


  std::vector<char*> av;

  void check_fw_equal(unsigned n_layers, bool should_be_equal){
    p_R = model.add_parameters({out_size , n_nodes});
    p_bias = model.add_parameters({out_size});

    builder_rnn = shared_ptr<dynet::LSTMBuilder>(new dynet::LSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
    builder_rnn->new_graph(cg);
    builder_rnn->start_new_sequence();
    i_wr_1 = input(cg, dynet::Dim({wordrep_size}), {-1,0,1});
    i_h_1 = builder_rnn->add_input(i_wr_1);
    i_wr_2 = input(cg, dynet::Dim({wordrep_size}), {-2,0,2});
    i_h_2 = builder_rnn->add_input(i_wr_2);
    i_wr_3 = input(cg, dynet::Dim({wordrep_size}), {-3,0,3});
    i_h_3 = builder_rnn->add_input(i_wr_3);
    i_wr_4 = input(cg, dynet::Dim({wordrep_size}), {-4,0,4});
    i_h_4 = builder_rnn->add_input(i_wr_4);
    R = parameter(cg, p_R); // hidden -> word rep parameter
    bias = parameter(cg, p_bias);  // word bias
    u_t = affine_transform({bias, R, i_h_4});
    dynet::expr::Expression loss_exp = pickneglogsoftmax(u_t, desired);
    e_rnn = cg.forward(loss_exp);




  //  Tensor e_rnn = cg.forward();

    std::shared_ptr<dynet::PyramidLSTMBuilder> builder_pyramid_;
    builder_pyramid_ = shared_ptr<dynet::PyramidLSTMBuilder>(new dynet::PyramidLSTMBuilder(n_layers, wordrep_size, n_nodes, &model));
    builder_pyramid_->params = builder_rnn->params;
    builder_pyramid_->new_graph(cg);
    builder_pyramid_->start_new_sequence();
    i_wr_1 = input(cg, dynet::Dim({wordrep_size}), {-1,0,1});
    i_h_1 = builder_pyramid_->add_input(i_wr_1);
    i_wr_2 = input(cg, dynet::Dim({wordrep_size}), {-2,0,2});
    i_h_2 = builder_pyramid_->add_input(i_wr_2);
    i_wr_3 = input(cg, dynet::Dim({wordrep_size}), {-3,0,3});
    i_h_3 = builder_pyramid_->add_input(i_wr_3);
    i_wr_4 = input(cg, dynet::Dim({wordrep_size}), {-4,0,4});
    i_h_4 = builder_pyramid_->add_input(i_wr_4);

    R = parameter(cg, p_R); // hidden -> word rep parameter
    bias = parameter(cg, p_bias);  // word bias
    u_t = affine_transform({bias, R, i_h_4});
    loss_exp = pickneglogsoftmax(u_t, desired);
    Tensor e_pyramid = cg.forward(loss_exp);

    if(should_be_equal) BOOST_CHECK_EQUAL(*e_rnn.v, *e_pyramid.v);
    else BOOST_CHECK_NE(*e_rnn.v, *e_pyramid.v);
  }

};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(pyramid_lstm_equality_test, PyramidLstmTest);




BOOST_AUTO_TEST_CASE( pyramid_1layer_equal ) {
  check_fw_equal(1, true);
}

BOOST_AUTO_TEST_CASE( pyramid_2layer_unequal ) {
  check_fw_equal(2, false);
}


BOOST_AUTO_TEST_CASE( pyramid_height_3_layers ) {
  PyramidLSTMBuilder builder(3, 1, 1, &model);
  BOOST_CHECK_EQUAL(builder.pyramid_height(0), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(1), 2);
  BOOST_CHECK_EQUAL(builder.pyramid_height(2), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(3), 3);
  BOOST_CHECK_EQUAL(builder.pyramid_height(4), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(5), 2);
  BOOST_CHECK_EQUAL(builder.pyramid_height(6), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(7), 3);
}

BOOST_AUTO_TEST_CASE( pyramid_height_1_layer ) {
  PyramidLSTMBuilder builder(1, 1, 1, &model);
  BOOST_CHECK_EQUAL(builder.pyramid_height(0), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(1), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(2), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(3), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(4), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(5), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(6), 1);
  BOOST_CHECK_EQUAL(builder.pyramid_height(7), 1);
}





BOOST_AUTO_TEST_SUITE_END()


