import nltk
import torch
import numpy as np
import gradio as gr
from nltk import sent_tokenize
from transformers import RobertaTokenizer, RobertaForMaskedLM

nltk.download('punkt')

cuda = torch.cuda.is_available()

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForMaskedLM.from_pretrained("roberta-large")
if cuda:
    model = model.cuda()

max_len = 20
top_k = 100
temperature = 1
burnin = 250
max_iter = 500


# adapted from https://github.com/nyu-dl/bert-gen
def generate_step(out,
                  gen_idx,
                  temperature=None,
                  top_k=0,
                  sample=False,
                  return_list=True):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k 
    """
    logits = out.logits[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1,
                             index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample()  # removed superfluous squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx


# adapted from https://github.com/nyu-dl/bert-gen
def parallel_sequential_generation(seed_text,
                                   seed_end_text,
                                   max_len=max_len,
                                   top_k=top_k,
                                   temperature=temperature,
                                   max_iter=max_iter,
                                   burnin=burnin):
    """ Generate for one random position at a timestep
    
    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
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
        out = model(**inp)
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(out,
                             gen_idx=seed_len + kk,
                             top_k=topk,
                             temperature=temperature,
                             sample=(ii < burnin))
        inp['input_ids'][0][seed_len + kk] = idxs[0]

    tokens = inp['input_ids'].cpu().numpy()[0][masked_tokens]
    tokens = tokens[(np.where((tokens != tokenizer.eos_token_id)
                              & (tokens != tokenizer.bos_token_id)))]
    return tokenizer.decode(tokens)


def inbertolate(doc,
                max_len=15,
                top_k=0,
                temperature=None,
                max_iter=300,
                burnin=200):
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
            new_doc += parallel_sequential_generation(para[sentence],
                                                      para[sentence + 1],
                                                      max_len=max_len,
                                                      top_k=top_k,
                                                      temperature=temperature,
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
                gr.Textbox(label="Text", lines=7),
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
                gr.Slider(label="Maximum iterations",
                          minimum=0,
                          maximum=1000,
                          value=max_iter),
                gr.Slider(label="Burn-in",
                          minimum=0,
                          maximum=500,
                          value=burnin),
            ],
            outputs=gr.Textbox(label="Expanded text", lines=24))
    block.launch(server_name='0.0.0.0')
