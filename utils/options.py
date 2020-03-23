# -*- coding: utf-8 -*-
import argparse

def parse_args_function():
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--input_file",
        default='./datasets/obman/',
        help="Input image, directory"
    )
    parser.add_argument(
        "--output_file",
        default='./checkpoints/model-',
        help="Prefix of output pkl filename"
    )
    # Optional arguments.
    parser.add_argument(
        "--train",
        action='store_true',
        help="Training mode."
    )
    parser.add_argument(
        "--val",
        action='store_true',
        help="Use validation set."
    )
    parser.add_argument(
        "--test",
        action='store_true',
        help="Test model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default = 512,
        help="Mini-batch size"
    )
    parser.add_argument(
        "--model_def",
        default='GraphUNet',
        help="Name of the model 'GraphUNet', 'GraphNet' or 'DenseNet'"
    )
    parser.add_argument(
        "--pretrained_model",
        default='',
        help="Load trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--gpu_number",
        type=int,
        nargs='+',
        default = [0],
        help="Identifies the GPU number to use."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default = 0.01,
        help="Identifies the optimizer learning rate."
    )
    parser.add_argument(
        "--lr_step",
        type=int,
        default = 1000,
        help="Identifies the adaptive learning rate step size."
    )
    parser.add_argument(
        "--lr_step_gamma",
        type=float,
        default = 0.1,
        help="Identifies the adaptive learning rate step gamma."
    )
    parser.add_argument(
        "--log_batch",
        type=int,
        default = 1000,
        help="Show log samples."
    )
    parser.add_argument(
        "--val_epoch",
        type=int,
        default = 1,
        help="Run validation on epochs."
    )
    parser.add_argument(
        "--snapshot_epoch",
        type=int,
        default = 10,
        help="Save snapshot epochs."
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default = 10000,
        help="Maximum number of epochs."
    )
    args = parser.parse_args()
    return args
