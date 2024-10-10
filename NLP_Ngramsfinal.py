import nltk
import itertools
import re
import math

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

def preprocess(corpus):
    corpus = [[word.lower() for word in sentence] for sentence in corpus]
    corpus = [[re.sub(r'[^\w\s]', '', word) for word in sentence] for sentence in corpus]
    lemmatizer = WordNetLemmatizer()
    corpus = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in corpus]

    return corpus

# Load Dataset
def load_corpus(file_path):
    with open(file_path, 'r') as f:
        corpus = f.readlines()
    return [sentence.strip().split() for sentence in corpus]

class UnigramModel:
    def __init__(self, corpus,k=0):
        self.corpus = corpus
        self.word_counts = defaultdict(int)
        self.total_words = 0
        self.vocab = set()
        self.unknown_token = "<UNK>"
        self.add_k = k  # Default value for Add-K smoothing
        self.vocab_size = 0
        self.build_unigram_model()

    def build_unigram_model(self):

        for word in self.corpus:
            if word not in self.vocab:
                self.vocab.add(word)
            self.word_counts[word] += 1
            self.total_words += 1

    def get_count(self, word):

        return self.word_counts.get(word, 0)  # Return 0 for unseen words

    def get_probability(self, word):

        count = self.get_count(word)
        self.vocab_size = len(self.vocab)
        return (count + self.add_k) / (self.total_words + self.add_k * self.vocab_size)


    def unigram_perplexity(self, test_corpus):

        log_prob_sum = 0
        test_words = len(test_corpus)  # Number of words in the test corpus

        for word in test_corpus:
            prob = self.get_probability(word)
            if prob == 0:  # Avoid log(0)
                prob = 1 / (self.total_words + self.add_k * self.vocab_size)# Assign a small probability
            log_prob_sum += math.log(prob)

        # Calculate perplexity
        perplexity = math.exp(-log_prob_sum / test_words)
        return perplexity

class BigramModel:
    def __init__(self, corpus,k):
        self.corpus = corpus
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.total_bigrams = 0
        self.vocab = set()
        self.add_k = k
        self.build_bigram_model()

    def build_bigram_model(self):

        for i in range(len(self.corpus) - 1):
            word1 = self.corpus[i]
            word2 = self.corpus[i + 1]
            self.vocab.add(word1)
            self.vocab.add(word2)
            self.bigram_counts[(word1, word2)] += 1
            self.unigram_counts[word1] += 1
            self.total_bigrams += 1

        # Don't forget the last unigram
        self.unigram_counts[self.corpus[-1]] += 1
        self.vocab.add(self.corpus[-1])

    def get_probability(self, word1, word2):
        bigram_count = self.bigram_counts.get((word1, word2), 0)
        firstword_count = self.unigram_counts.get(word1, 0)  # Count of the first word in the bigram
        vocab_size = len(self.vocab)
        return (bigram_count + self.add_k) / (firstword_count + self.add_k * vocab_size) if firstword_count > 0 else 0


    def Bigram_perplexity(self, test_corpus):

        log_prob_sum = 0
        test_bigrams = len(test_corpus) - 1

        for i in range(test_bigrams):
            word1 = test_corpus[i]
            word2 = test_corpus[i + 1]
            prob = self.get_probability(word1, word2)

            if prob == 0:  # Avoid log(0)
                prob = 1e-10  # Assign small probability for unseen bigrams

            log_prob_sum += math.log(prob)

        perplexity = math.exp(-log_prob_sum / test_bigrams)
        return perplexity


# Function to evaluate the Bigram model


if __name__ == "__main__":

    file_path = 'train.txt'  # Path to the training text file
    test_path = 'test.txt'  # Path to the test text file

    # Load and Preprocess the Corpus
    train_data = load_corpus(file_path)
    test_data = load_corpus(test_path)

    train_corpus = preprocess(train_data)
    test_corpus = preprocess(test_data)

    # Flatten the corpus (list of lists -> list of words)
    train_corpus_flat = list(itertools.chain(*train_corpus))
    test_corpus_flat = list(itertools.chain(*test_corpus))


    # Split training data into train and validation sets
    train_data, validation_data = train_test_split(train_data, train_size=0.9, random_state=192)
    val_corpus = preprocess(validation_data)
    val_corpus_flat = list(itertools.chain(*val_corpus))
    # print(train_corpus)
    # Train unigram and bigram models
    for val in [0,0.5,0.4]:
        unigram_model = UnigramModel(train_corpus_flat,val)  # Using Laplace smoothing with k=1

        bigram_model = BigramModel(train_corpus_flat,val)  # Using Laplace smoothing with k=1

        print(f"\nN-gram perplexities for k = {val}\n")
        # Calculate perplexity on the validation set
        unigram_perplexity =  unigram_model.unigram_perplexity(val_corpus_flat)
        print(f"Unigram Model Perplexity: {unigram_perplexity:.4f}")

        bigram_perplexity = bigram_model.Bigram_perplexity(val_corpus_flat)
        print(f"Bigram Model Perplexity: {bigram_perplexity:.4f}")

        # Calculate perplexity on the test set
        unigram_perplexity_test = unigram_model.unigram_perplexity(test_corpus_flat)
        print(f"Unigram Model Perplexity (Test Set): {unigram_perplexity_test:.4f}")

        bigram_perplexity_test = bigram_model.Bigram_perplexity(test_corpus_flat)
        print(f"Bigram Model Perplexity (Test Set): {bigram_perplexity_test:.4f}")
