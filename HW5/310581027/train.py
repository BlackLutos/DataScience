# %% [markdown]
# ### Load libraries

# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm
import torch.utils.data as data
import pandas as pd
import torch
import transformers

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# %% [markdown]
# ### Load tokenizer and model

# %%
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# %% [markdown]
# ### Data processing

# %%
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("json", data_files="train.json")
body_text = dataset['train']['body']
title_text = dataset['train']['title']
input_text = []
summary = []
data_size = 100000
for body in range(data_size):
    input_text.append('summarize: ' + str(body_text[body]))
for title in range(data_size):
    summary.append(str(title_text[title]))

input_train, input_val, summary_train, summary_val = train_test_split(input_text, summary, test_size=0.1, random_state=42)


# %% [markdown]
# ### Build datasets

# %%
class HeadlineGenerationDataset(Dataset):
    def __init__(self, text, summary, tokenizer, max_len = 512):
        self.data = []
        for t, s in zip(text, summary):
            input_t = tokenizer(t, truncation=True, padding="max_length", max_length=max_len)
            label_t = tokenizer(s, truncation=True, padding="max_length", max_length=max_len)

            #轉換-100
            for cnt,tmp in enumerate(label_t['input_ids']):
                if tmp == 0:
                    label_t['input_ids'][cnt] = -100
                    
            self.data.append({'input_ids':torch.tensor(input_t['input_ids']),
                              'attention_mask':torch.tensor(input_t['attention_mask']),
                              'labels':torch.tensor(label_t['input_ids'])})

    def __getitem__(self, index):
        
         
        return self.data[index]
        

    def __len__(self):
        return len(self.data)

# %% [markdown]
# ### Dataloader

# %%
train_set = HeadlineGenerationDataset(input_train, summary_train, tokenizer,max_len = 512)
train_loader = DataLoader(train_set,batch_size = 2,shuffle = True, num_workers = 0, pin_memory = True)
val_set = HeadlineGenerationDataset(input_val, summary_val, tokenizer,max_len = 512)
val_loader = DataLoader(val_set,batch_size = 2,shuffle = True, num_workers = 0, pin_memory = True)

# %% [markdown]
# ### Valdation model

# %%
def valdation(val_loader, model, tokenizer):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            for i in range(len(outputs)):
                output = tokenizer.decode(outputs[i], skip_special_tokens=True)
                preds.append(output)
                label = labels[i].cpu().numpy()
                label = label[label != -100]  # filter out padding labels
                targets.append(tokenizer.decode(label, skip_special_tokens=True))
    return preds, targets


# %% [markdown]
# ### Evaluation

# %%
import evaluate

def evaluation(outputs, targets):
    metric_rouge = evaluate.load("rouge", rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"])
    rouge = metric_rouge.compute(predictions=outputs, references=targets, use_stemmer=True)
    return rouge

# %% [markdown]
# ### Train model

# %%
import os

model.cuda()
optimizer = torch.optim.AdamW(params = model.parameters(), lr = 1e-4)
if os.path.exists('checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for epoch in range(10):
    model.train()
    train = tqdm(train_loader)
    for data in train:
        for key in data.keys():
            data[key] = data[key].cuda()
        outputs = model(**data)
        loss = outputs.loss
        train.set_description(f'Epoch {epoch+1}')
        train.set_postfix({'Loss': loss.item()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, 'checkpoint.pth')
    outputs, targets = valdation(val_loader, model, tokenizer)
    rouge = evaluation(outputs, targets)
    print("Rouge scores: " , rouge)
        
    
    model.save_pretrained('model_{}'.format(epoch+1))


