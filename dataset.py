import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, training_data, training_label, pad_token_id):
        self.training_data = training_data
        self.training_label = training_label
        self.pad_token_id = pad_token_id
        assert self.training_data.shape == self.training_label.shape
        
    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        input_ids = self.training_data[idx]
        return {"input_ids": input_ids, "labels": self.training_label[idx], "attention_mask": input_ids.ne(self.pad_token_id)}

if __name__ == "__main__":

    from transformers import AutoModelForCausalLM
    from tqdm import tqdm
    
    training_data = torch.load(f'ultrachat_input_ids.pt', map_location="cpu")
    training_label = torch.load(f'ultrachat_labels.pt', map_location="cpu")
    
    dataset = TextDataset(training_data, training_label, pad_token_id=2)
    dataloader = DataLoader(dataset, batch_size=4)

    model_name = "HuggingFaceH4/zephyr-7b-beta"
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(dtype).cuda()

    with torch.no_grad(): 
        for j, inputs in tqdm(enumerate(dataloader)):
            input_ids = inputs["input_ids"].cuda()
            labels = inputs["labels"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            print(output.loss)
