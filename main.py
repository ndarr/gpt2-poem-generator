import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data.dataloader import DataLoader
from poem_dataset import PoemDataset
import os
import tqdm
from random import randint

path = "data/sonnet/"
sonnet_files = [path + f for f in os.listdir(path)]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True)
model.resize_token_embeddings(len(tokenizer))
model.cuda()
optimizer = torch.optim.AdamW(params=model.parameters())

poem_data = PoemDataset(sonnet_files)
data_loader_params = {
    "batch_size": 3,
    "shuffle": False
}
data_loader = DataLoader(poem_data, **data_loader_params)

epochs = 30

losses = []
for i in range(epochs):
    tqdm_loader = tqdm.tqdm(data_loader)
    for poems in tqdm_loader:
        optimizer.zero_grad()
        inputs = tokenizer(poems, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses = [loss.cpu().item()] + losses
        if len(losses) > 100:
            losses = losses[:100]
        tqdm_loader.set_description("Loss %f" % (sum(losses)/len(losses)))

generated_sequences = []
with torch.no_grad():
    for i in range(50):
        print("########## Sampling ##########")
        # Sample first token random
        rand_indices = [randint(0, len(tokenizer)) for _ in range(10)]

        output_sequences = model.generate(
            input_ids=torch.tensor(rand_indices, device=torch.device("cuda")).view(10, 1),
            max_length=60,
            temperature=0.7,
            no_repeat_ngram_size=3,
            early_stopping=True)
        print(output_sequences.shape)
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()
            # Decode text
            text = tokenizer.decode(generated_sequence[1:], clean_up_tokenization_spaces=True).strip()
            # text = text.split("<eos>")[0]
            # text = text.split("[PAD]")[0]
            # text = text.replace("<eol> ", "\n")

            generated_sequences.append(text)
            print("########## Finished ##########")

with open("gpt2_poems.txt", "w+") as f:
    for seq in generated_sequences:
        f.write(seq)
        f.write("\n\n")


