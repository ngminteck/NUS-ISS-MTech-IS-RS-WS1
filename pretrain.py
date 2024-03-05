from . import encoder
from . import model
from . import framework
import torch
import os
import sys
import json
import numpy as np
import logging
import requests
from pathlib import Path

root_url = "https://thunlp.oss-cn-qingdao.aliyuncs.com/"
default_root_path = os.path.join(os.getcwd(), '.opennre').replace('\\','/')

def check_root(root_path=default_root_path):
   
    directory_check = []
    directory_check.append(root_path)
    directory_check.append(os.path.join(root_path, 'benchmark').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'pretrain').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'benchmark/fewrel').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'benchmark/fewrel').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'benchmark/nyt10').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'benchmark/nyt10m').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'benchmark/semeval').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'benchmark/wiki20m').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'benchmark/wiki80').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'benchmark/tacred').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'benchmark/wiki_distant').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'pretrain/glove').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'pretrain/bert-base-uncased').replace('\\','/'))
    directory_check.append(os.path.join(root_path, 'pretrain/nre').replace('\\','/'))
    
    for dir in directory_check: 
        if not os.path.exists(dir):
            os.mkdir(dir)

def download_file(dest_folder,additional_url):
    dest_folder = dest_folder.replace('\\','/')
    url = root_url + additional_url
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_folder,filename).replace('\\','/')
    my_file = Path(filepath)
    
    if my_file.is_file():
        print(filename,"existed in",dest_folder)
    else:
        print("Downloading",url,"...")
        response = requests.get(url, stream = True)
        if response.status_code != 200:
            print("Unable to get response from",url,response.status_code)
        else:
            file_path = os.path.join(dest_folder, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                f.close()
            print(url,"download completed.")

def download_wiki80(root_path=default_root_path):  
    dest_folder = os.path.join(root_path, 'benchmark/wiki80')
    download_file(dest_folder,'opennre/benchmark/wiki80/wiki80_rel2id.json')
    download_file(dest_folder,'opennre/benchmark/wiki80/wiki80_train.txt')
    download_file(dest_folder,'opennre/benchmark/wiki80/wiki80_val.txt')

def download_tacred(root_path=default_root_path):
    dest_folder = os.path.join(root_path, 'benchmark/tacred')
    download_file(dest_folder,'opennre/benchmark/tacred/tacred_rel2id.json')
    logging.info('Due to copyright limits, we only provide rel2id for TACRED. Please download TACRED manually and convert the data to OpenNRE format if needed.')

def download_nyt10(root_path=default_root_path):
    dest_folder = os.path.join(root_path, 'benchmark/nyt10')
    download_file(dest_folder,'opennre/benchmark/nyt10/nyt10_rel2id.json')
    download_file(dest_folder,'opennre/benchmark/nyt10/nyt10_train.txt')
    download_file(dest_folder,'opennre/benchmark/nyt10/nyt10_test.txt')

def download_nyt10m(root_path=default_root_path):
    dest_folder = os.path.join(root_path, 'benchmark/nyt10m')
    download_file(dest_folder,'opennre/benchmark/nyt10m/nyt10m_rel2id.json')
    download_file(dest_folder,'opennre/benchmark/nyt10m/nyt10m_train.txt')
    download_file(dest_folder,'opennre/benchmark/nyt10m/nyt10m_test.txt')
    download_file(dest_folder,'opennre/benchmark/nyt10m/nyt10m_val.txt')
        
def download_wiki20m(root_path=default_root_path):
    dest_folder = os.path.join(root_path, 'benchmark/wiki20m')
    download_file(dest_folder,'opennre/benchmark/wiki20m/wiki20m_rel2id.json')
    download_file(dest_folder,'opennre/benchmark/wiki20m/wiki20m_train.txt')
    download_file(dest_folder,'opennre/benchmark/wiki20m/wiki20m_test.txt')
    download_file(dest_folder,'opennre/benchmark/wiki20m/wiki20m_val.txt')

def download_wiki_distant(root_path=default_root_path):
    dest_folder = os.path.join(root_path, 'benchmark/wiki_distant')
    download_file(dest_folder,'opennre/benchmark/wiki_distant/wiki_distant_rel2id.json')
    download_file(dest_folder,'opennre/benchmark/wiki_distant/wiki_distant_train.txt')
    download_file(dest_folder,'opennre/benchmark/wiki_distant/wiki_distant_test.txt')
    download_file(dest_folder,'opennre/benchmark/wiki_distant/wiki_distant_val.txt')

def download_semeval(root_path=default_root_path):
    dest_folder = os.path.join(root_path, 'benchmark/semeval')
    download_file(dest_folder,'opennre/benchmark/semeval/semeval_rel2id.json')
    download_file(dest_folder,'opennre/benchmark/semeval/semeval_train.txt')
    download_file(dest_folder,'opennre/benchmark/semeval/semeval_test.txt')
    download_file(dest_folder,'opennre/benchmark/semeval/semeval_val.txt')

def download_glove(root_path=default_root_path):
    dest_folder = os.path.join(root_path, 'pretrain/glove')
    download_file(dest_folder,'opennre/pretrain/glove/glove.6B.50d_mat.npy')
    download_file(dest_folder,'opennre/pretrain/glove/glove.6B.50d_word2id.json')
    
def download_bert_base_uncased(root_path=default_root_path):
    dest_folder = os.path.join(root_path, 'pretrain/bert-base-uncased')
    download_file(dest_folder,'opennre/pretrain/bert-base-uncased/config.json')
    download_file(dest_folder,'opennre/pretrain/bert-base-uncased/pytorch_model.bin')
    download_file(dest_folder,'opennre/pretrain/bert-base-uncased/vocab.txt')

def download_pretrain(model_name, root_path=default_root_path):
    dest_folder = os.path.join(root_path, 'pretrain/nre/')
    download_file(dest_folder, 'opennre/pretrain/nre/' + model_name + '.pth.tar')

def download(name, root_path=default_root_path):
    if not os.path.exists(os.path.join(root_path, 'benchmark')):
        os.mkdir(os.path.join(root_path, 'benchmark'))
    if not os.path.exists(os.path.join(root_path, 'pretrain')):
        os.mkdir(os.path.join(root_path, 'pretrain'))
    if name == 'nyt10':
        download_nyt10(root_path=root_path)
    elif name == 'nyt10m':
        download_nyt10m(root_path=root_path)
    elif name == 'wiki20m':
        download_wiki20m(root_path=root_path)
    elif name == 'wiki_distant':
        download_wiki_distant(root_path=root_path)
    elif name == 'semeval':
        download_semeval(root_path=root_path)
    elif name == 'wiki80':
        download_wiki80(root_path=root_path)
    elif name == 'tacred':
        download_tacred(root_path=root_path)
    elif name == 'glove':
        download_glove(root_path=root_path)
    elif name == 'bert_base_uncased':
        download_bert_base_uncased(root_path=root_path)
    else:
        raise Exception('Cannot find corresponding data.')

def get_model(model_name, root_path=default_root_path):
    check_root()
    ckpt = os.path.join(root_path, 'pretrain/nre/' + model_name + '.pth.tar').replace('\\','/')
    if model_name == 'wiki80_cnn_softmax':
        print("Downloading required file, please wait.")
        download_pretrain(model_name, root_path=root_path)
        download('glove', root_path=root_path)
        download('wiki80', root_path=root_path)
        print("All required file have been downloaded.")
        wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json').replace('\\','/')))
        word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy').replace('\\','/'))
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json').replace('\\','/')))
        sentence_encoder = encoder.CNNEncoder(token2id=wordi2d,
                                                     max_length=40,
                                                     word_size=50,
                                                     position_size=5,
                                                     hidden_size=230,
                                                     blank_padding=True,
                                                     kernel_size=3,
                                                     padding_size=1,
                                                     word2vec=word2vec,
                                                     dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['wiki80_bert_softmax', 'wiki80_bertentity_softmax']:
        print("Downloading required file, please wait.")
        download_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('wiki80', root_path=root_path)
        print("All required file have been downloaded.")
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json').replace('\\','/')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased').replace('\\','/'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased').replace('\\','/'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['tacred_bert_softmax', 'tacred_bertentity_softmax']:
        print("Downloading required file, please wait.")
        download_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('tacred', root_path=root_path)
        print("All required file have been downloaded.")
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/tacred/tacred_rel2id.json').replace('\\','/')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased').replace('\\','/'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased').replace('\\','/'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    else:
        raise NotImplementedError