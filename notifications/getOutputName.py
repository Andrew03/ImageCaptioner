import sys
sys.path.append(".")
import param_parser
import file_namer

params = param_parser.parse_train_params(sys.argv[1])[:-1]
print(file_namer.make_output_name(*params, isTrain=True))
print(file_namer.make_output_name(*params, isTrain=False))
