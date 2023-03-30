# SeeNear sentiment analysis

Our model was developed using Pytorch and KOBERT. 

## Test sentiment analysis

This model shows the maximum two emotions(in 6 emotions) detected and the percentage of the responses to one conversation.
Number of emotions are six(happy, embarrassed, anger, anxiety, heartbroken, sad).

If you want to use it, you can use the [model](https://huggingface.co/hyeok212/sentiment) as follows.  

```python
from flask import Flask, request

app = Flask(__name__)

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd

#KoBERT
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
#transformer
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sentiment_repository.kobertm import BERTDataset
from sentiment_repository.kobertm import BERTClassifier

## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#bertmodel의 vocabulary
device = torch.device("cpu")
    
bertmodel, vocab = get_pytorch_kobert_model()

model = BERTClassifier(bertmodel).to(device)
model.load_state_dict(torch.load('model.pt', map_location='cpu'))

@app.route("/predict/<arg>")
def predict(arg):
    predict_sentence = arg
    
    #토큰화
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    def new_softmax(a) : 
        c = np.max(a) # 최댓값
        exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
        sum_exp_a = np.sum(exp_a)
        y = (exp_a / sum_exp_a) * 100
        return np.round(y, 3)


    def ping(arg):
        data = [predict_sentence, '0']
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)
    
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)

            valid_length= valid_length
            label = label.long().to(device)
        
            out = model(token_ids, valid_length, segment_ids)
        

            test_eval=[]
            for i in out:
                logits=i
                logits = logits.detach().cpu().numpy()
                min_v = min(logits)
                total = 0
                probability = []
                result_emotion = []
                percent = []
                logits = np.round(new_softmax(logits), 3).tolist()
                for logit in logits:
                    print(logit)
                    probability.append(np.round(logit, 3))


            if np.argmax(logits) == 0:  emotion = "0"
            elif np.argmax(logits) == 1: emotion = "1"
            elif np.argmax(logits) == 2: emotion = '2'
            elif np.argmax(logits) == 3: emotion = '3'
            elif np.argmax(logits) == 4: emotion = '4'
            elif np.argmax(logits) == 5: emotion = '5'

            result_emotion.append(emotion)
            percent.append(probability[np.argmax(logits)])
            logits[np.argmax(logits)] = 0

            if np.argmax(logits) == 0:  emotion = "0"
            elif np.argmax(logits) == 1: emotion = "1"
            elif np.argmax(logits) == 2: emotion = '2'
            elif np.argmax(logits) == 3: emotion = '3'
            elif np.argmax(logits) == 4: emotion = '4'
            elif np.argmax(logits) == 5: emotion = '5'
     
            result_emotion.append(emotion)
            percent.append(probability[np.argmax(logits)])
            print(result_emotion, percent)

        return result_emotion, percent
    
    result_emotion, percent = ping(predict_sentence)

    if percent[0] <= 60.0:
        return 'null'    
    
    
    #emotion_list = '0':'happy','1':'embarrassed','2':'anger','3':'anxiety','4':'heartbroken','5':'sad'
    else:
        return dict({'result_emotion': result_emotion, 'percent': percent})



```

### Reference
1. SKT-Brain KOBERT model
https://github.com/SKTBrain/KoBERT

2. Dataset    
[Korean emotional conversation corpus](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)  
[Continuous Conversation Dataset with Korean Sentiment Information](https://aihub.or.kr/aihubdata/data/view.do?)
