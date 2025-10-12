"""
Evaluate a model on either perplexity (for language modeling) or accuracy (for classification)

Perplexity eval code originally written by Yegor Kuznetsov for UW CSE 447.
"""
import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from bpe import BPETokenizer, pad_to_length
import numpy as np
from model import GPT
from multi_head_attention import init_qkv_proj, self_attention


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():  # only works on macOS
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"PyTorch version: {torch.__version__} on {DEVICE}")

def get_arguments():

    parser = argparse.ArgumentParser(description="Given a saved model, evaluate either perplexity or classification accuracy")

    parser.add_argument("-m", help="modelfile: the name/path of the model to load after training using train.py",
                        default='best.pretrain.model')

    parser.add_argument("-i",  help="inputfile: the name/path of the test file that has to be read one text per line",
                        default='datasets/1b_benchmark.dev.tokens')

    parser.add_argument("-o", help="outputfile: the name/path of the output file to write predictions (ignored for LM task)",
                        default='outputs/outputs.txt')
    
    parser.add_argument("-t", help="task: 'classification' to evaluate classification accuracy, 'lm' to evaluate perplexity",
                        default='lm')
    parser.add_argument("-l", help="'inline' if the labels are on each line of the input file after a <TAB>, or path to label file (ignored for LM task)", default=None)
    parser.add_argument("-d", action="store_true", help="pass this flag to eval perplexity on dummy data (use for debugging). works for LM task only.", default=False)
    parser.add_argument("-q", action="store_true", help="pass this flag to suppress tqdm progress bars", default=False)

    return parser.parse_args()

# autograder will call your eval_classification and eval_perplexity functions
# you shouldn't need to change these unless you want to
@torch.no_grad
def eval_classification(data, model, tokenizer, gold_labels=None, bs=32, progress=True, pad_to_len=100):
    """ 
        Evaluate classification on a dataset.

        pass in a list of gold labels to get accuracy against them, or just get predictions for unlabelled data
    """
    it = range(0, len(data), bs)
    if progress: it = tqdm(it)

    tokenizer = BPETokenizer()
    
    preds = []
    for b_start in it:
        batch_data = data[b_start:b_start + bs]

        tokens = torch.tensor(
            [pad_to_length(tokenizer(t).squeeze(0).tolist(), pad_to_len, tokenizer.pad_id) for t in batch_data],
            dtype=torch.long).to(DEVICE)
        preds += model.classify(tokens)

    if gold_labels is not None:
        preds = torch.tensor(preds)
        acc = (preds == gold_labels).float().mean().item()
    else:
        acc = None

    return preds, acc

# if you change the padding max length, you may need to update the default value
# assumes data does not include classification labels - if it does, you should strip them off before calling this
@torch.no_grad
def eval_perplexity(data, model, tokenizer, bs=32, progress=True, pad_to_len=100):
    """ 
        Evaluate perplexity on a dataset.
        This implementation originally by Yegor Kuznetsov.
    """
    it = range(0, len(data), bs)
    if progress: it = tqdm(it)
    
    out = []
    for b_start in it:
        batch_data = data[b_start:b_start + bs]
        tokens = torch.tensor(
            [pad_to_length(tokenizer(t).squeeze(0).tolist(), pad_to_len, tokenizer.pad_id) for t in batch_data],
            dtype=torch.long).to(DEVICE)
        X_tokens, y_tokens = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

        model.eval()
        logits, _, _, _ = model(X_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        y_log_probs = torch.gather(log_probs, 2, y_tokens[..., None])[..., 0]

        for i in range(y_tokens.shape[0]):
            not_pad = (y_tokens[i] != tokenizer.pad_id)
            loss = -y_log_probs[i, not_pad].mean()
            out.append(loss.item())

    # return mean perplexity over dataset
    return np.mean(np.exp(out))

def main(args):
    # load model
    tokenizer = BPETokenizer()
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.pad_token = tokenizer.pad_id
    model_config.model_type = 'gpt-nano' # change this or add a command line arg if desired
    model_config.vocab_size = max(tokenizer.encoder.encoder.values()) + 1 # +1 to accomodate PAD token

    # The model's context length
    # Note that minGPT has learned posemb, so outside the used maxlen wont really work
    model_config.block_size = 1024

    # Use the attention function you implemented in the last part
    model_config.attn_init_fn = init_qkv_proj # we implemented this for you
    model_config.attn_fn = self_attention # you implemented this

    # try to guess number of classes from model name
    num_classes = 0
    if args.t == "classification":
        if 'products' in args.m:
            num_classes = 2
        elif '4dim' in args.m:
            num_classes = 4
        elif 'questions' in args.m:
            num_classes = 6
        elif 'news' in args.m:
            num_classes = 9
        else:
            print("unable to guess number of classes from model name.")
            exit(1)

    model_config.num_classes = num_classes

    model = GPT(model_config)
    model.load_state_dict(torch.load(args.m, map_location=DEVICE))

    if args.d:
      data = ["this is a dummy sentence."] * 10
    else:
        with open(args.i) as f:
            data = [line.strip() for line in f if line.strip()]

    # run eval
    if args.t == 'classification':
        if args.l == 'inline':
            # split labels from data
            data, labels = map(list, zip(*(s.split('\t') for s in data)))
        elif os.path.exists(args.l):
            # labels are in a file
            with open(args.l, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
        else:
            print("trying to evaluate classification accuracy, but no labels provided.")
            exit(1)

        unique_labels = sorted(set(labels))  # sort for consistency
        label2id = {label: idx for idx, label in enumerate(unique_labels)}

        labels = torch.tensor([label2id[label] for label in labels], dtype=torch.long) # map label to id
        preds, acc = eval_classification(data, model, tokenizer, gold_labels=labels, progress=(not args.q))
        # Save the predictions: one label prediction per line
        with open(args.o, "w") as file:
            for pred in preds:
                file.write(str(pred.item())+"\n")

        # print out accuracy
        print(f"Accuracy: {acc:.3f}")
        return acc
    
    else:
        # evaluate perplexity
        perplexity = eval_perplexity(data, model, tokenizer, progress=(not args.q))
        # print out accuracy
        print(f"Perplexity: {perplexity:.5f}")
        return perplexity

if __name__ == "__main__":
    args = get_arguments()
    main(args)
