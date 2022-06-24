import os
import re
import json
import pandas as pd
import gradio as gr
import numpy as np

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# dataset 

def load_dataset(dataset, threshold):
    directory = os.path.join("NLP", "cluster_0.3") 
        
    ds = {} 
    ds['entity'] = torch.load(directory+'/entity.pt', map_location=torch.device('cpu')) # list(string)
    ds['embed'] = torch.load(directory+'/embed.pt', map_location=torch.device('cpu')) # list(array)
    ds['pairs'] = torch.load(directory+'/pairs.pt', map_location=torch.device('cpu')) # list(list(int))
    ds['pair_paper'] = torch.load(directory+'/pair_paper.pt', map_location=torch.device('cpu')) # dict(str:dict)
    return ds
    
 # model

def text_preprocessing(text):
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocessing_for_bert(entities):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    # For every sentence...
    for sent in entities:
        whole_sent = "This paper uses " + str(sent) + "."
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(whole_sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=16,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            truncation=True,
            padding='max_length'
            )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

class Model(nn.Module):
    def __init__(self, freeze_bert):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.b = nn.Parameter(torch.normal(0, 1, size=(768, 1))) 
        self.W = nn.Parameter(torch.normal(0, 1, size=(768, 768)))
        self.c = nn.Parameter(torch.zeros(1))

    def get_embedding(self, inputs, masks): # output the bert model embedding
        h = self.bert(input_ids=inputs, attention_mask=masks)[0][:,0,:]
        return h
    
    def scoring(self, x):
        # x is the combination of concept features
        # \mean_i (A_i)*b
        first_term_value = torch.mean(torch.stack([i @ self.b for i in x]))
        # \mean_ij A_i.T * W_ij * A_j
        pairwise_comb = torch.combinations(torch.arange(0,len(x),dtype=torch.long))
        second_term_value = torch.mean(torch.stack([x[i] @ self.W @ x[j] for i,j in pairwise_comb]))
        return torch.sigmoid(first_term_value + second_term_value + self.c)
    
def load_model(dataset, threshold):
    # directory = os.path.join(dataset,threshold,"model.pt")
    directory = os.path.join("NLP", "cluster_0.3","sci_disco_model_nlp_0.2.pt")
    model = Model(False)
    # load pre-trained model
    checkpoint = torch.load(directory, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    return model
    
# input one concept
def one_input(entity, dataset, model, display_num):
    # get entity embed
    input_ids, attention_masks = preprocessing_for_bert(entity)
    # scoring
    entities = dataset['entity']
    embed = dataset['embed']
    pairs = dataset['pairs']
    pair_paper = dataset['pair_paper']
    output = model.get_embedding(input_ids, attention_masks).reshape(-1)
    
    score = [model.scoring([output, embed[i], embed[j]]).detach().item() for i,j in pairs]
    score = np.array(score)
    
    indices = np.argsort(score)[-display_num:][::-1]
    
    df = {}
    df['score'] = [str(round(score[i], 3)) for i in indices]
    df['input'] = [entity] * display_num
    df['entity1 in cropus'] = [entities[pairs[i][0]] for i in indices]
    df['entity2 in cropus'] = [entities[pairs[i][1]] for i in indices]
    df['title'] = [pair_paper[str(pairs[i][0])+" and "+str(pairs[i][1])]['title'] for i in indices]
    df['year'] = [pair_paper[str(pairs[i][0])+" and "+str(pairs[i][1])]['year'] for i in indices]
    df['conf'] = [pair_paper[str(pairs[i][0])+" and "+str(pairs[i][1])]['conf'] for i in indices]
    
    df = pd.DataFrame(data=df)
    return df
    
# input two concept
def two_input(entity, dataset, model, display_num):
    entities = dataset['entity']
    embed = dataset['embed']
    input_ids, attention_masks = preprocessing_for_bert(entity)
    output = model.get_embedding(input_ids, attention_masks)
    score = [model.scoring([output[0], output[1], e]).detach().item() for e in embed]
    score = np.array(score)
    
    indices = np.argsort(score)[-display_num:][::-1]
    
    df = {}
    df['score'] = [str(round(score[i], 3)) for i in indices]
    df['input1'] = [entity[0]] * display_num
    df['input2'] = [entity[1]] * display_num
    df['entity in cropus'] = [entities[i] for i in indices]
    
    df = pd.DataFrame(data=df)
    return df
    
# input three concept
def three_input(entity, model):
    input_ids, attention_masks = preprocessing_for_bert(entity)
    output = model.get_embedding(input_ids, attention_masks)
    score = model.scoring([output[0], output[1], output[2]]).detach().item()
    df = {}
    df['score'] = [str(round(score, 3))]
    df['input1'] = [entity[0]]
    df['input2'] = [entity[1]]
    df['input3'] = [entity[2]]
    
    df = pd.DataFrame(data=df)
    return df
    
# main function
def example_fn(text, display_num, dataset, threshold):
    if not text: 
        return pd.DataFrame(data={"Error!": ["Please enter 1-3 concepts"]})
    
    # check the input type
    concept_list = text.split("\n")
    
    if len(concept_list) > 3:
        return pd.DataFrame(data={"Error!": ["Please enter 1-3 concepts"]})
    
    # load dataset 
    ds = load_dataset(dataset, threshold) # a. entity name b. entity embedding c. pair_comb d. pair_paper
    
    # load model
    model = load_model(dataset, threshold)
    
    if len(concept_list) == 1:
        df = one_input(concept_list, ds, model, display_num)
    elif len(concept_list) == 2:
        df = two_input(concept_list, ds, model, display_num)
    else:
        df = three_input(concept_list, model)
    
    '''
    d = {'score': [0.7], 'c1': ["input aggregation"], 'c2': ['sentence level autoencoder'], 
         'c3': ["statistical machine translation"], 
         'title': ["Combination of Arabic Preprocessing Schemes for Statistical Machine Translation"], 
         'conference': ["ACL"], 'year': [2006]}
    df = pd.DataFrame(data=d)
    '''
    return df
    

demo = gr.Interface(
    fn = example_fn,
    inputs = [gr.Textbox(label="Input concepts, seperate by line", lines=3, ), 
              gr.Slider(10, 50, step=1, label="Display Number", value=20),
              gr.Radio(["NLP", "CV", "NLP+CV"], label="Dataset", value="NLP"),
              gr.Radio(["0.2", "0.3", "0.5"], label="Threshold", value="0.3")
             ],
    outputs = [gr.DataFrame(label="Output", row_count=1, 
                            headers=["score","...","title","conference","year"])],
    title="Scentific Discovery",
    allow_flagging="never",
    examples = [["summarization", 20, "NLP", "0.2"],
                ["scene graph", 30, "CV", "0.3"]],
)

demo.launch()
