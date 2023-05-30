# %% [markdown]
# ### Load libraries

# %%
import os
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm
import torch.utils.data as data
import pandas as pd
import torch
import transformers

# %% [markdown]
# ### Load tokenizer and model

# %%
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# %% [markdown]
# ### Data processing

# %%
from datasets import load_dataset
dataset = load_dataset("json", data_files="test.json")
body_text = dataset['train']['body']
input_text = []
for body in range(13762):
    input_text.append('summarize: ' + str(body_text[body]))


# %% [markdown]
# ### Build Dataset

# %%
class HeadlineGenerationDataset(Dataset):
    def __init__(self, text, tokenizer, max_len = 512):
        self.data = []
        for t in text:
            input_t = tokenizer(t, truncation=True, padding="max_length", max_length=max_len)
            
                    
            self.data.append({'input_ids':torch.tensor(input_t['input_ids']),
                              'attention_mask':torch.tensor(input_t['attention_mask'])})

    def __getitem__(self, index):
        
         
        return self.data[index]
        

    def __len__(self):
        return len(self.data)

# %% [markdown]
# ### Dataloader

# %%
test_set = HeadlineGenerationDataset(input_text, tokenizer,max_len = 512)
test_loader = DataLoader(test_set,batch_size = 6,shuffle = False, num_workers = 0, pin_memory = True)

# %% [markdown]
# ### Test model

# %%
import os
import json

model.cuda()
model.load_state_dict(torch.load('model_best/pytorch_model.bin'))
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        for i in range(len(outputs)):
            output = tokenizer.decode(outputs[i], skip_special_tokens=True)
            print(output)
            print('------------------------')
            print('------------------------')
            print('------------------------')
            predicted_text = {"title": output}
            ans_json = json.dumps(predicted_text, separators=(',', ':'))
            with open('310581027.json', 'a') as f:
                f.write(ans_json + "\n")


