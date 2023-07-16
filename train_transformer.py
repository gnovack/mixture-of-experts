import argparse
import transformers
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig, TrainingArguments
from transformers.training_args import TrainingArguments
from trainer import TransformerTrainer
from utils import human_readable, get_device, load_dataset

def get_parameter_count(model):
    return sum([p.numel() for p in model.parameters()])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, choices=["75m"])
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    args = parser.parse_args()

    if args.model_size == "75m":
        hidden_size = 768
        sequence_length = 1024
        num_layers = 5

    batch_size = 42
    gradient_accumulation_steps = 3

    model_config = OPTConfig(
        hidden_size=hidden_size,
        ffn_dim=4*hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=hidden_size // 64,
        max_position_embeddings=sequence_length,
    )
    model = OPTForCausalLM(model_config)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

    print("Loading Training Dataset...")
    train_dataset = load_dataset(tokenizer, training_datasets=["c4", "wikipedia", "openwebtext"], max_length=sequence_length)
    
    parameter_count = get_parameter_count(model)

    print("=" * 20, "Training Model", "=" * 20)
    print("Parameter count:", human_readable(parameter_count))

    checkpoint_path = f"./checkpoints/dense-{args.model_size}"

    max_train_tokens = 1e9
    max_train_steps = int(max_train_tokens // (batch_size * gradient_accumulation_steps * sequence_length))
    print(f"Training for {max_train_steps:,} steps")

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1e-4,
        lr_scheduler_type=transformers.SchedulerType.COSINE,
        max_steps=max_train_steps,
        warmup_steps=300, # TODO
        use_mps_device=get_device() == "mps",
        logging_steps=args.checkpoint_steps,
        save_steps=args.checkpoint_steps,
        fp16=get_device() == "cuda",
        save_total_limit=3
    )

    trainer = TransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    transformers.logging.set_verbosity_info()
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()
