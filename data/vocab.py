import pickle

class Vocab(object):
    def __init__(self, logs):

        self.stoi = {}
        self.itos = []
        for line in logs:
            self.itos.append(line)
        self.itos = list(set(self.itos))
        print(f"Vocab size: {len(self.itos)}")
        self.unk_index = len(self.itos)+1
        self.stoi = {e: i+1 for i, e in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def get_event(self, real_event):
        event = self.stoi.get(real_event, self.unk_index)
        return event

    def save_vocab(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    @staticmethod
    def load_vocab(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
