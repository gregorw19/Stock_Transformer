import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tmodel import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import warnings

from tqdm import tqdm

from pathlib import Path

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # don't have decoder, change so it works
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target(decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output of decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:,-1])
        # Get token with max probability (bc its a greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

# data should be (batch, seq_len, features)
def tensorize(data):
    # make a tensor out of inputted data, will take training data, and return training data split into groups of size determined in config
    tensor = torch.Tensor(data)
    return tensor

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    # size of control window
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # print to console
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            # is for testing
            if count == num_examples:
                break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) 
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_input = np.array(pd.read_csv("train.csv"))
    ds_ans = np.array(pd.read_csv("answers.csv"))

    # split data 90/10 for training
    train_ds_size = int(0.9 * len(ds_input))
    val_ds_size = len(ds_input) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids =  tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids =  tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config):
    model = build_transformer(config['seq_len'], config['d_model'], config['features'])
    return model

def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok = True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['prelaod'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            model.train()

            encoder_input = batch['encoder_input'].to(device) # (Batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (Batch, 1, 1, seq_len) (only hides padding tokens)
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, seq_len, Seq_len) (hides subsequent words and padding, that's why additional seq_len)

            # run tensors through transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (Batch, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (Batch, seq_len)

            # (Batch, seq_len, tgt_vocab_size) -> (Batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # backprop
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch, 
            'model_state': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)


