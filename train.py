import argparse
import pickle
from model import GPT
from trainer import Trainer
from bpe import BPETokenizer, pad_to_length
from multi_head_attention import init_qkv_proj, self_attention
import torch
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():  # only works on macOS
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"PyTorch version: {torch.__version__} on {DEVICE}")


def get_arguments():
    parser = argparse.ArgumentParser(description="MinGPT trainer wrapper")
    parser.add_argument("-t", help="task: 'pretrain' (LM only) or 'finetune' (also trains classification head)",
                        default='pretrain')
    parser.add_argument("-i", help="path of the input file where training file is in the form <text>TAB<label> for finetuning, or text separated by newlines for pretraining",
                        default='datasets/1b_benchmark.train.tokens')
    parser.add_argument("-p", help="path of pretrained model (if not starting from scratch)",
                        default=None)
    parser.add_argument("-o", help="path of the file where the model is saved", default='best.pretrain.model')
    parser.add_argument("-d", action="store_true", help="pass this flag to train on a small dummy input (use for debugging). works for LM task only.", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    
    tokenizer = BPETokenizer()

    if args.d:
      lines = ["this is a dummy sentence."] * 1000
    else:
        with open(args.i) as f:
            lines = [line.strip() for line in f if line.strip()]

    if args.t == "finetune":
        # do finetuning
        # do tokenization

        for i, line in enumerate(lines):
            if '\t' not in line:
                print(i)
        labels = [line.split('\t')[1] for line in lines]
        unique_labels = sorted(set(labels))  # sort for consistency
        label2id = {label: idx for idx, label in enumerate(unique_labels)}

        labels = torch.tensor([label2id[line.split('\t')[1]] for line in lines], dtype=torch.long) # map label to id
        tokenized = [tokenizer(line.split('\t')[0]) for line in lines]  # tokenize only the text part
        num_classes = len(label2id)

    else:
        # do pretraining - language modeling only
        # do tokenization
        tokenized = [tokenizer(line)[0] for line in lines]  # [0] removes batch dim
        labels = None
        num_classes = 0
    
    # pad sequences to same length
    max_len = 100  # we have not tuned this - you are encouraged to experiment
    padded = torch.tensor([pad_to_length(t.squeeze(0).tolist(), max_len, tokenizer.pad_id) for t in tokenized], dtype=torch.long)
        
    # set up model and Trainer
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.pad_token = tokenizer.pad_id

    # This configuration is the very small 'gpt-nano' defined in minGPT. we'd use a
    # bigger model like 'gpt2' but it would take a very long time to train :(
    # See model.py for configurations of other models
    model_config.model_type = 'gpt-nano'
    # 'gpt-nano' equivalent to:
    # model_config.n_layer = 3
    # model_config.n_head = 3
    # model_config.n_embd = 48

    model_config.vocab_size = max(tokenizer.encoder.encoder.values()) + 1 # +1 to accomodate PAD token

    # The model's context length
    # Note that minGPT has learned posemb, so outside the used maxlen wont really work
    model_config.block_size = 1024

    # Use the attention function you implemented in the last part
    model_config.attn_init_fn = init_qkv_proj # we implemented this for you
    model_config.attn_fn = self_attention # you implemented this

    # handle num classes for classification
    # will init a new classification head if the pretrained model doesn't have one
    model_config.num_classes = num_classes

    model = GPT(model_config)

    if args.p is not None:
        # strict=False to retain default initialization for (possibly new) classification head
        model.load_state_dict(torch.load(args.p, map_location=DEVICE), strict=False) 
    train_config = Trainer.get_default_config()
    train_config.device = DEVICE
    train_config.num_workers = 2

    # We didn't tune the hyperparameters at all, please experiment with these!
    train_config.learning_rate = 5e-4
    train_config.batch_size = 32
    # TODO you should probably increase this
    train_config.max_iters = len(tokenized) // train_config.batch_size  # train for 1 epoch
    # train_config.max_iters = 1 # uncomment this for quick debugging

    trainer = Trainer(train_config, model, padded, labels)

    # run training
    model.to(DEVICE)
    model.train()

    bar = tqdm(total=train_config.max_iters)
        
    @torch.no_grad()
    def on_batch_end(trainer):
        if trainer.classification_loss is not None and trainer.labels is not None:
            preds = trainer.classification_logits.argmax(dim=-1)
            acc = (preds == trainer.batch_labels).float().mean().item()
            bar.set_postfix(accuracy=acc)
        else:
            bar.set_postfix(lm_loss=trainer.lm_loss.item())
        bar.update()

    trainer.set_callback('on_batch_end', on_batch_end)
    trainer.run()
    bar.close()

    torch.save(model.state_dict(), args.o)





