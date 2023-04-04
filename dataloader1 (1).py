class Dataloader:

    def __init__(self, path):
        self.path = path
        #print("Path:", self.path)
    
    def loader(self):
        f = open(self.path, "r", encoding="utf-8") #"Line_labels.txt"
        words_list = []
        for line in f:
            line_split = line.strip().split('\t')
            words_list.append(line)
        return words_list

    ## Dataset splitting
    ## We will split the dataset into three subsets with a 90:5:5 ratio (train:validation:test).
    def datasplit(self, words_list):
        
        split_idx = int(0.90 * len(words_list))
        train_samples = words_list[:split_idx]
        validation_samples = words_list[split_idx:]
        #val_split_idx = int(0.2 * len(test_samples))
        #validation_samples = test_samples[:val_split_idx]
        #test_samples = test_samples[val_split_idx:]
        assert len(words_list) == len(train_samples) + len(validation_samples) #+ len(
                #test_samples)

        #print(f"Total training samples: {len(train_samples)}")
        #print(f"Total validation samples: {len(validation_samples)}")
        #print(f"Total test samples: {len(test_samples)}")

        return train_samples, validation_samples#, test_samples

    ## Data input pipeline
    ## We start building our data input pipeline by first preparing the image paths.
    def get_image_paths_and_labels(self, samples):
        paths = []
        labels = []
        for sample in samples:
            line_split = sample.strip().split('\t')
            paths.append(line_split[0])
            labels.append(str(line_split[1]))
        return paths, labels

    def train_clean_labels(self, train_labels):
        # Find maximum length and the size of the vocabulary in the training data.
        train_labels_cleaned = []
        characters = set()
        max_len = 0

        for label in train_labels:
            #label = label.split(" ")[-1].strip()
            for char in label:
                characters.add(char)

            max_len = max(max_len, len(label))
            train_labels_cleaned.append(label)

        #print("Maximum length: ", max_len)
        #print("Vocab size: ", len(characters))

        return train_labels_cleaned, max_len, characters

    def clean_labels(self, labels):
        cleaned_labels = []
        for label in labels:
            #label = label.split(" ")[-1].strip()
            cleaned_labels.append(label)
        return cleaned_labels