from typing import Dict
from transformers import Trainer


class TransformerTrainer(Trainer):
    
    def log(self, logs: Dict[str, float]):

        effective_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        sequence_length = self.model.config.max_position_embeddings

        flops_per_step = float(
            estimate_transformer_flops(self.model.config.hidden_size, sequence_length) * 
            self.model.config.num_hidden_layers *
            effective_batch_size *
            3  # backward is roughly 2x forward
        )

        logs['tokens_seen'] = self.state.global_step * effective_batch_size * sequence_length
        logs['flops'] = self.state.global_step * flops_per_step
        return super().log(logs)


def estimate_transformer_flops(hidden_size, sequence_length):
    return 24 * sequence_length * hidden_size**2 + 4 * sequence_length**2 * hidden_size