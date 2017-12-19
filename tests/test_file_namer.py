import unittest
import test_helper
import sys
sys.path.append(".")
import os
import file_namer

class TestFileNamer(unittest.TestCase):

  def setUp(self):
    self.checkpoints = ("tests/model_batch_10_occurrence_5_epoch_10_dropout_0.0_dim_512x512_clip_5_isnorm",
        "tests/model_batch_10_occurrence_5_epoch_1_dropout_0.0_dim_512x512_clip_5_nonorm")
    for checkpoint in self.checkpoints:
      checkpoint_file = open(checkpoint, "w")
      checkpoint_file.close()
    
  def test_make_vocab_name(self):
    expected = "data/vocab/vocab_occurrence_5.txt"
    test_helper.test_equal(self, expected, file_namer.make_vocab_name,
      "Incorrect Vocab Name", 5)

  def test_make_batch_name(self):
    expected = "data/batched_data/train_batch_10_occurrence_5.txt"
    test_helper.test_equal(self, expected, file_namer.make_batch_name,
      "Incorrect Train Batch Name", 10, 5, True)
    expected = "data/batched_data/val_batch_10_occurrence_5.txt"
    test_helper.test_equal(self, expected, file_namer.make_batch_name,
      "Incorrect Val Batch Name", 10, 5, False)

  def test_make_output_name(self):
    expected = "output/train_batch_10_occurrence_5_epoch_10_dropout_0.0_decoderLR_0.001_encoderLR_0.001_dim_512x512_clip_5_nonorm.txt"
    test_helper.test_equal(self, expected, file_namer.make_output_name,
      "Incorrect Train Output Not Normalized Name", 10, 5, 10, 0.0, 0.001, 0.001, 512, 512, 5, False, True)
    expected = "output/val_batch_10_occurrence_5_epoch_10_dropout_0.0_decoderLR_0.001_encoderLR_0.001_dim_512x512_clip_5_isnorm.txt"
    test_helper.test_equal(self, expected, file_namer.make_output_name,
      "Incorrect Val Output Normalized Name", 10, 5, 10, 0.0, 0.001, 0.001, 512, 512, 5, True, False)

  def test_make_checkpoint_name(self):
    expected = "checkpoints/model_batch_10_occurrence_5_epoch_10_dropout_0.0_decoderLR_0.001_encoderLR_0.001_dim_512x512_clip_5_nonorm.pt"
    test_helper.test_equal(self, expected, file_namer.make_checkpoint_name,
      "Incorrect Checkpoint Name", 
      10, 5, 10, 0.0, 0.001, 0.001, 512, 512, 5, False)

  def test_is_checkpoint(self):
    for checkpoint in self.checkpoints:
      self.assertTrue(file_namer.is_checkpoint(checkpoint))
  
  def test_get_checkpoint(self):
    test_helper.test_equal(self, self.checkpoints[0], file_namer.get_checkpoint,
      "Failed to Find Checkpoint",
      self.checkpoints[0])
    test_helper.test_equal(self, self.checkpoints[0], file_namer.get_checkpoint,
      "Failed to Find Checkpoint",
      self.checkpoints[0].replace("epoch_10", "epoch_15"))

  def tearDown(self):
    for checkpoint in self.checkpoints:
      os.remove(checkpoint)

suite = unittest.TestLoader().loadTestsFromTestCase(TestFileNamer)
unittest.TextTestRunner(verbosity=2).run(suite)
