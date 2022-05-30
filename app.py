import nltk
import torch
import numpy as np
import gradio as gr
from nltk import sent_tokenize

from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM,
    LogitsProcessorList,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
)
from transformers.generation_logits_process import TypicalLogitsWarper

nltk.download('punkt')

cuda = torch.cuda.is_available()

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForMaskedLM.from_pretrained("roberta-large")
if cuda:
    model = model.cuda()

max_len = 20
top_k = 100
temperature = 1
typical_p = 0
burnin = 250
max_iter = 500


# adapted from https://github.com/nyu-dl/bert-gen
def generate_step(out: object,
                  gen_idx: int,
                  top_k: int = top_k,
                  temperature: float = temperature,
                  typical_p: float = typical_p,
                  sample: bool = False) -> list:
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - temperature (float): sampling temperature
        - typical_p (float): if >0 use typical sampling
        - sample (bool): if True, sample from full distribution.
    
    returns:
        - list: batch_size tokens
    """
    logits = out.logits[:, gen_idx]
    logit_warpers = []
    if top_k > 0:
        logit_warpers += [TopKLogitsWarper(top_k)]
    if temperature:
        logit_warpers += [TemperatureLogitsWarper(temperature)]
    if typical_p > 0:
        if typical_p >= 1:
            typical_p = 0.999
        logit_warpers += [TypicalLogitsWarper(typical_p)]
    logits = LogitsProcessorList(logit_warpers)(None, logits)

    if sample:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_tokens = torch.argmax(logits, dim=-1)

    return next_tokens.tolist()


# adapted from https://github.com/nyu-dl/bert-gen
def parallel_sequential_generation(seed_text: str,
                                   seed_end_text: str,
                                   max_len: int = max_len,
                                   top_k: int = top_k,
                                   temperature: float = temperature,
                                   typical_p: float = typical_p,
                                   max_iter: int = max_iter,
                                   burnin: int = burnin) -> str:
    """ Generate text consistent with preceding and following text
    
    Args:
        - seed_text (str): preceding text
        - seed_end_text (str): following text
        - top_k (int): if >0, only sample from the top k most probable words
        - temperature (float): sampling temperature
        - typical_p (float): if >0 use typical sampling
        - max_iter (int): number of iterations in MCMC
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax

    Returns:
        - string: generated text to insert between seed_text and seed_end_text
    """
    inp = tokenizer(seed_text + tokenizer.mask_token * max_len + seed_end_text,
                    return_tensors='pt')
    masked_tokens = np.where(
        inp['input_ids'][0].numpy() == tokenizer.mask_token_id)[0]
    seed_len = masked_tokens[0]
    if cuda:
        inp = inp.to('cuda')

    for ii in range(max_iter):
        kk = np.random.randint(0, max_len)
        idxs = generate_step(model(**inp),
                             gen_idx=seed_len + kk,
                             top_k=top_k if (ii >= burnin) else 0,
                             temperature=temperature,
                             typical_p=typical_p,
                             sample=(ii < burnin))
        inp['input_ids'][0][seed_len + kk] = idxs[0]

    tokens = inp['input_ids'].cpu().numpy()[0][masked_tokens]
    tokens = tokens[(np.where((tokens != tokenizer.eos_token_id)
                              & (tokens != tokenizer.bos_token_id)))]
    return tokenizer.decode(tokens)


def inbertolate(doc: str,
                max_len: int = max_len,
                top_k: int = top_k,
                temperature: float = temperature,
                typical_p: float = typical_p,
                max_iter: int = max_iter,
                burnin: int = burnin):
    """ Pad out document generating every other sentence
    
    Args:
        - doc (str): document text
        - max_len (int): number of tokens to insert between sentences
        - top_k (int): if >0, only sample from the top k most probable words
        - temperature (float): sampling temperature
        - typical_p (float): if >0 use typical sampling
        - max_iter (int): number of iterations in MCMC
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax

    Returns:
        - string: generated text to insert between seed_text and seed_end_text
    """
    new_doc = ''
    paras = doc.split('\n')

    for para in paras:
        para = sent_tokenize(para)
        if para == '':
            new_doc += '\n'
            continue
        para += ['']

        for sentence in range(len(para) - 1):
            new_doc += para[sentence] + ' '
            new_doc += parallel_sequential_generation(
                para[sentence],
                para[sentence + 1],
                max_len=max_len,
                top_k=top_k,
                temperature=float(temperature),
                typical_p=typical_p,
                burnin=burnin,
                max_iter=max_iter) + ' '

        new_doc += '\n'
    return new_doc


if __name__ == '__main__':
    block = gr.Blocks(css='.container')
    with block:
        gr.Markdown("<h1><center>inBERTolate</center></h1>")
        gr.Markdown(
            "<center>Hit your word count by using BERT to pad out your essays!</center>"
        )
        gr.Interface(
            fn=inbertolate,
            inputs=[
                gr.Textbox(label="Text", lines=10),
                gr.Slider(label="Maximum length to insert between sentences",
                          minimum=1,
                          maximum=40,
                          step=1,
                          value=max_len),
                gr.Slider(label="Top k", minimum=0, maximum=200, value=top_k),
                gr.Slider(label="Temperature",
                          minimum=0,
                          maximum=2,
                          value=temperature),
                gr.Slider(label="Typical p",
                          minimum=0,
                          maximum=1,
                          value=typical_p),
                gr.Slider(label="Maximum iterations",
                          minimum=0,
                          maximum=1000,
                          value=max_iter),
                gr.Slider(label="Burn-in",
                          minimum=0,
                          maximum=500,
                          value=burnin),
            ],
            outputs=gr.Textbox(label="Expanded text", lines=30))
    block.launch(server_name='0.0.0.0')
