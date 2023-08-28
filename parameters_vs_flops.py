import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from math import floor, log
from tabulate import tabulate
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTConfig


def initialze_transformer_layer(hidden_size: int):
    config = OPTConfig(
            hidden_size=hidden_size,
            ffn_dim=4*hidden_size,
            num_hidden_layers=1,
            num_attention_heads=hidden_size // 64,
        )
    return OPTDecoderLayer(config)

def get_parameter_count(model):
    return sum([p.numel() for p in model.parameters()])

def estimate_parameter_count(hidden_size):
    return 12*hidden_size**2 + 13*hidden_size

def human_readable(number):
    """Thanks to https://stackoverflow.com/a/45478574/13731609"""
    units = ['', 'K', 'M', 'B', 'T']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

def get_flops(model, sequence_length: int, hidden_size: int):
    flops, _, _ = get_model_profile(
        model,
        input_shape=(1, sequence_length, hidden_size),
        print_profile=False,
        as_string=False,
        detailed=False
    )
    return flops

def estimate_flops(hidden_size, sequence_length):
    return 24 * sequence_length * hidden_size**2 + 4 * sequence_length**2 * hidden_size

if __name__ == "__main__":

    hidden_sizes = [256, 512, 768, 1024, 2048, 4096, 8192, 16384]
    
    estimated_parameter_counts = [
        human_readable(estimate_parameter_count(hidden_size))
        for hidden_size in hidden_sizes
    ]
    
    parameter_counts = [
        human_readable(get_parameter_count(initialze_transformer_layer(hidden_size)))
        for hidden_size in hidden_sizes
    ]

    table = zip(hidden_sizes, estimated_parameter_counts, parameter_counts)
    table_headers=["Hidden Size", "Parameter Count (est.)", "Parameter Count"]

    if torch.cuda.is_available():
        estimated_flops = [
            human_readable(estimate_flops(hidden_size, 1024))
            for hidden_size in hidden_sizes
        ]
        table_headers.append("Forward Pass FLOPs (est.)")

        flops = [
            human_readable(get_flops(initialze_transformer_layer(hidden_size), 1024, hidden_size))
            for hidden_size in hidden_sizes
        ]
        table_headers.append("Forward Pass FLOPs")
        table = zip(hidden_sizes, estimated_parameter_counts, parameter_counts, estimated_flops, flops)
    else:
        print("No CUDA device available, skipping FLOPS computation.")

    print(tabulate(
        table,
        headers=table_headers,
        tablefmt="fancy_grid"
    ))


# Output:
# ╒═══════════════╤══════════════════════════╤═══════════════════╤═════════════════════════════╤══════════════════════╕
# │   Hidden Size │ Parameter Count (est.)   │ Parameter Count   │ Forward Pass FLOPs (est.)   │ Forward Pass FLOPs   │
# ╞═══════════════╪══════════════════════════╪═══════════════════╪═════════════════════════════╪══════════════════════╡
# │           256 │ 789.76K                  │ 789.76K           │ 2.68B                       │ 2.69B                │
# ├───────────────┼──────────────────────────┼───────────────────┼─────────────────────────────┼──────────────────────┤
# │           512 │ 3.15M                    │ 3.15M             │ 8.59B                       │ 8.61B                │
# ├───────────────┼──────────────────────────┼───────────────────┼─────────────────────────────┼──────────────────────┤
# │           768 │ 7.09M                    │ 7.09M             │ 17.72B                      │ 17.74B               │
# ├───────────────┼──────────────────────────┼───────────────────┼─────────────────────────────┼──────────────────────┤
# │          1024 │ 12.60M                   │ 12.60M            │ 30.06B                      │ 30.10B               │
# ├───────────────┼──────────────────────────┼───────────────────┼─────────────────────────────┼──────────────────────┤
# │          2048 │ 50.36M                   │ 50.36M            │ 111.67B                     │ 111.73B              │
# ├───────────────┼──────────────────────────┼───────────────────┼─────────────────────────────┼──────────────────────┤
# │          4096 │ 201.38M                  │ 201.38M           │ 429.50B                     │ 429.62B              │
# ├───────────────┼──────────────────────────┼───────────────────┼─────────────────────────────┼──────────────────────┤
# │          8192 │ 805.41M                  │ 805.41M           │ 1.68T                       │ 1.68T                │
# ├───────────────┼──────────────────────────┼───────────────────┼─────────────────────────────┼──────────────────────┤
# │         16384 │ 3.22B                    │ 3.22B             │ 6.67T                       │ 6.67T                │
# ╘═══════════════╧══════════════════════════╧═══════════════════╧═════════════════════════════╧══════════════════════╛