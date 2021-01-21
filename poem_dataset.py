from torch.utils.data.dataset import Dataset


class PoemDataset(Dataset):
    def __init__(self, poem_files):
        self.poem_files = poem_files
        self.poems = self.preload_poems()

    def __getitem__(self, index):
        return self.poems[index]

    def preload_poems(self):
        poems = []
        for poem_file in self.poem_files:
            poem = open(poem_file, "r").readline().strip()
            poem = poem.replace("<eol> ", "\n")
            poems.append(poem)
        return poems

    def __len__(self):
        return len(self.poems)
