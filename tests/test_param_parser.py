import unittest
import test_helper
import sys
sys.path.append(".")
import os
import param_parser

class TestParamParser(unittest.TestCase):
  def setUp(self):
    self.create_train_param_files()
    self.create_run_param_files()

  def test_parse_train_params(self):
    expected = (5, 10, 512, 512, 0.0, 0.001, 0.001, 10, 5, True, True)
    test_helper.test_equal(self, expected, param_parser.parse_train_params,
      "Incorrect Train Params (Use Cuda, With Norm)", 'train_use_cuda_is_norm.txt')
    expected = (5, 10, 512, 512, 0.0, 0.001, 0.001, 10, 5, False, False)
    test_helper.test_equal(self, expected, param_parser.parse_train_params,
      "Incorrect Train Params (No Cuda, No Norm)", 'train_no_cuda_no_norm.txt')

  def test_parse_run_params(self):
    expected = (5, 10, 512, 512, 0.0, 0.001, 0.001, 10, 5, 5, 10, True, True, True)
    test_helper.test_equal(self, expected, param_parser.parse_run_params,
      "Incorrect Run Params (Use Cuda, With Norm)", 'run_use_cuda_is_norm.txt')
    expected = (5, 10, 512, 512, 0.0, 0.001, 0.001, 10, 5, 5, 10, False, False, False)
    test_helper.test_equal(self, expected, param_parser.parse_run_params,
      "Incorrect Run Params (No Cuda, No Norm)", 'run_no_cuda_no_norm.txt')

  def tearDown(self):
    os.remove('train_use_cuda_is_norm.txt')
    os.remove('train_no_cuda_no_norm.txt')
    os.remove('run_use_cuda_is_norm.txt')
    os.remove('run_no_cuda_no_norm.txt')

  def create_train_param_files(self):
    train_use_cuda_is_norm = open("train_use_cuda_is_norm.txt","w")
    train_no_cuda_no_norm = open("train_no_cuda_no_norm.txt","w")
    for test_file in (train_use_cuda_is_norm, train_no_cuda_no_norm):
      test_file.write("min_occurrences: 5 \n")
      test_file.write("batch_size: 10 \n")
      test_file.write("embedding_dim: 512 \n")
      test_file.write("hidden_size: 512 \n")
      test_file.write("dropout: 0.0 \n")
      test_file.write("decoder_lr: 0.001 \n")
      test_file.write("encoder_lr: 0.001 \n")
      test_file.write("num_epochs: 10 \n")
      test_file.write("grad_clip: 5 \n")
    train_use_cuda_is_norm.write("isNormalized: True \n")
    train_use_cuda_is_norm.write("useCuda: True \n")
    train_no_cuda_no_norm.write("isNormalized: False \n")
    train_no_cuda_no_norm.write("useCuda: False \n")
    train_use_cuda_is_norm.close()
    train_no_cuda_no_norm.close()

  def create_run_param_files(self):
    run_use_cuda_is_norm = open("run_use_cuda_is_norm.txt","w")
    run_no_cuda_no_norm = open("run_no_cuda_no_norm.txt","w")
    for test_file in (run_use_cuda_is_norm, run_no_cuda_no_norm):
      test_file.write("min_occurrences: 5 \n")
      test_file.write("batch_size: 10 \n")
      test_file.write("embedding_dim: 512 \n")
      test_file.write("hidden_size: 512 \n")
      test_file.write("dropout: 0.0 \n")
      test_file.write("decoder_lr: 0.001 \n")
      test_file.write("encoder_lr: 0.001 \n")
      test_file.write("num_epochs: 10 \n")
      test_file.write("grad_clip: 5 \n")
      test_file.write("num_runs: 5 \n")
      test_file.write("beam_size: 10 \n")
    run_use_cuda_is_norm.write("isNormalized: True \n")
    run_use_cuda_is_norm.write("useCuda: True \n")
    run_use_cuda_is_norm.write("printStepProb: True \n")
    run_no_cuda_no_norm.write("isNormalized: False \n")
    run_no_cuda_no_norm.write("useCuda: False \n")
    run_no_cuda_no_norm.write("printStepProb: False \n")
    run_use_cuda_is_norm.close()
    run_no_cuda_no_norm.close()

suite = unittest.TestLoader().loadTestsFromTestCase(TestParamParser)
unittest.TextTestRunner(verbosity=2).run(suite)
