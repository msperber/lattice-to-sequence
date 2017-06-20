#pragma once

#include <lamtram/sentence.h>
#include <dynet/tensor.h>
#include <dynet/mp.h>
#include <boost/program_options.hpp>
#include <string>

namespace dynet {
struct Trainer;
class Model;
class Dict;
}

namespace lamtram {

class EvalMeasure;
class LLStats;

class LamtramTrain {

public:
    LamtramTrain() { }
    int main(int argc, char** argv);
    
    void TrainLM();
    void TrainEncDec();
    void TrainEncAtt();
    void TrainLattAtt();
    void TrainFeatAtt();
    void TrainEncCls();

    // Bilingual maximum likelihood training
    template<class ModelType, class OutputType>
    void BilingualTraining(const std::vector<Sentence> & train_src,
                           const std::vector<OutputType> & train_trg,
                           const std::vector<OutputType> & train_cache,
                           const std::vector<float> & train_weights,
                           const std::vector<Sentence> & dev_src,
                           const std::vector<OutputType> & dev_trg,
                           const dynet::Dict & vocab_src,
                           const dynet::Dict & vocab_trg,
                           dynet::Model & mod,
                           ModelType & encdec);

    // Bilingual maximum likelihood training
    template<class ModelType, class OutputType>
    void BilingualTrainingMP(const std::vector<Sentence> & train_src,
                           const std::vector<OutputType> & train_trg,
                           const std::vector<OutputType> & train_cache,
                           const std::vector<Sentence> & dev_src,
                           const std::vector<OutputType> & dev_trg,
                           const dynet::Dict & vocab_src,
                           const dynet::Dict & vocab_trg,
                           dynet::Model & mod,
                           ModelType & encdec);

    // Minimum risk training
    template<class ModelType>
    void MinRiskTraining(const std::vector<Sentence> & train_src,
                         const std::vector<Sentence> & train_trg,
                         const std::vector<int> & train_fold_ids,                         
                         const std::vector<Sentence> & dev_src,
                         const std::vector<Sentence> & dev_trg,
                         const dynet::Dict & vocab_src,
                         const dynet::Dict & vocab_trg,
                         const EvalMeasure & eval,
                         dynet::Model & model,
                         ModelType & encdec);

    // Get the trainer to use
    typedef std::shared_ptr<dynet::Trainer> TrainerPtr;
    TrainerPtr GetTrainer(const std::string & trainer_id, const dynet::real learning_rate, dynet::Model & model);

    // Load in the training data
    void LoadFile(const std::string filename, bool add_last, dynet::Dict & vocab, std::vector<Sentence> & sents);
    void LoadFileLattice(const std::string filename, dynet::Dict & vocab, std::vector<Sentence> & sents);
    void LoadFileFeatures(const std::string filename, std::vector<Sentence> & sents);
    void LoadLabels(const std::string filename, dynet::Dict & vocab, std::vector<int> & labs);
    void LoadWeights(const std::string filename, std::vector<float> & weights);

    void LoadBothFiles(
          const std::string filename_src, dynet::Dict & vocab_src, std::vector<Sentence> & sents_src,
          const std::string filename_trg, dynet::Dict & vocab_trg, std::vector<Sentence> & sents_trg);

protected:

    boost::program_options::variables_map vm_;

    // Variable settings
    dynet::real rate_thresh_, rate_decay_;
    int epochs_, context_, eval_every_;
    float scheduled_samp_, dropout_;
    std::string model_in_file_, model_out_file_;
    std::vector<std::string> train_files_trg_, train_files_src_, train_files_weights_;
    std::string dev_file_trg_, dev_file_src_;
    std::string softmax_sig_;

    std::vector<std::string> wildcards_;

};

template<class ModelType, class OutputType>
class BilingualLearner : public dynet::mp::ILearner<int, LLStats> {
public:
  explicit BilingualLearner(ModelType& encdec,
                            std::vector<std::vector<Sentence> >& train_src_minibatch,
                            std::vector<std::vector<OutputType> >& train_trg_minibatch,
                            std::vector<std::vector<OutputType> >& train_cache_minibatch,
                            std::vector<std::vector<Sentence> >& dev_src_minibatch,
                            std::vector<std::vector<OutputType> >& dev_trg_minibatch,
                            std::vector<std::vector<float> >& train_weights_minibatch,
                            std::vector<std::vector<float> >& dev_weights_minibatch,
                            dynet::real learning_rate, dynet::real learning_scale,
                            float scheduled_samp, std::string softmax_sig, int num_procs,
                            std::string model_out_file, std::string model_out_file_train_best,
                            const dynet::Dict & vocab_src, const dynet::Dict & vocab_trg, dynet::Model & model,
                            LamtramTrain::TrainerPtr trainer,
			    float dropout)
	    : encdec_(encdec), train_src_minibatch_(train_src_minibatch), train_trg_minibatch_(train_trg_minibatch), train_cache_minibatch_(train_cache_minibatch),
	      dev_src_minibatch_(dev_src_minibatch), dev_trg_minibatch_(dev_trg_minibatch),
          train_weights_minibatch_(train_weights_minibatch), dev_weights_minibatch_(dev_weights_minibatch),
	      do_dev_(dev_src_minibatch.size()>0), learning_rate_(learning_rate), learning_scale_(learning_scale),
	      scheduled_samp_(scheduled_samp), softmax_sig_(softmax_sig), epoch_frac_(0), num_procs_(num_procs), //train_ll_(vocab_trg.size()), dev_ll_(vocab_trg.size()),
	      model_out_file_(model_out_file), model_out_file_train_best_(model_out_file_train_best),
	      vocab_src_(vocab_src), vocab_trg_(vocab_trg), model_(model), trainer_(trainer), dropout_(dropout) {
  }
  virtual ~BilingualLearner() {}

  LLStats LearnFromDatum(const int& datum, bool learn) override ;

  virtual void StartTrain() override;
  virtual void StartDev() override;
  virtual void SaveModelDevBest() override ;
  virtual void SaveModelTrainBest() override ;

  virtual int GetNumSentForDatum(int datum) override ;
  virtual int GetNumWordsForDatum(int datum) override ;

  virtual void ReportStep(unsigned cid, dynet::mp::WorkloadHeader& header, unsigned num_sent, unsigned sent_delta,
			  unsigned num_words, unsigned words_delta, double time, double time_delta,
			  LLStats loss_delta) override ;

  virtual void ReportEvalPoint(bool dev, double fractional_iter, LLStats total_loss, bool new_best) override ;

private:
  ModelType& encdec_;
  std::vector<std::vector<Sentence> > & train_src_minibatch_;
  std::vector<std::vector<OutputType> > & train_trg_minibatch_;
  std::vector<std::vector<OutputType> > & train_cache_minibatch_;
  std::vector<std::vector<Sentence> > & dev_src_minibatch_;
  std::vector<std::vector<OutputType> > & dev_trg_minibatch_;
  std::vector<std::vector<float>> & train_weights_minibatch_;
  std::vector<std::vector<float>> & dev_weights_minibatch_;
  bool do_dev_;
  dynet::real learning_rate_, learning_scale_;
  float scheduled_samp_;
  std::string softmax_sig_;
  float epoch_frac_;
  float num_procs_;
  std::vector<OutputType> empty_cache_;
  std::string model_out_file_, model_out_file_train_best_;
  const dynet::Dict & vocab_src_;
  const dynet::Dict & vocab_trg_;
  dynet::Model & model_;
  LamtramTrain::TrainerPtr trainer_;
  float dropout_;
};



}
