import sys
sys.path.append(".")
import param_parser
import file_namer

params = param_parser.parse_train_params(sys.argv[1])[:-1]
min_occurrences = params[0]
batch_size = params[1]
embedding_dim = params[2]
hidden_size = params[3]
dropout = params[4]
decoder_lr = params[5]
encoder_lr = params[6]
num_epochs = params[7]
grad_clip = params[8]
isNormalized = params[9]
print(file_namer.make_output_name(batch_size, min_occurrences, num_epochs, dropout,
  decoder_lr, encoder_lr, embedding_dim, hidden_size, grad_clip, isNormalized, isTrain=True))
print(file_namer.make_output_name(batch_size, min_occurrences, num_epochs, dropout,
  decoder_lr, encoder_lr, embedding_dim, hidden_size, grad_clip, isNormalized, isTrain=False))
