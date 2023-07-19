from collections import defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = " ".join(pair)
    replacer = "".join(pair)
    for word in v_in:
        v_out[word.replace(bigram, replacer)] = v_in[word]
    return v_out

def bpe_segment(text, num_iters):
    vocab = defaultdict(int)
    for word in text.split():
        vocab[" ".join(word) + " </w>"] += 1
    
    for i in range(num_iters):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
    
    segments = " ".join(text.split())
    for segment in vocab:
        segments = segments.replace(segment, segment.replace(" ", "_"))
    
    return segments.split()

text = "Natural language processing (NLP) is a subfield of artificial intelligence and linguistics that focuses on the interactions between computers and human language. It involves developing algorithms and models that enable computers to understand, analyze, and generate human language in a useful and meaningful way. NLP has a wide range of applications, including machine translation, sentiment analysis, information retrieval, text summarization, and chatbots. BPE (Byte Pair Encoding) is a popular subword tokenization method used in NLP to handle complex word structures, handle out-of-vocabulary words, and improve language modeling tasks. It iteratively merges the most frequent pairs of characters or character sequences to build a vocabulary of subword units. The resulting subword units can better capture the morphological and semantic information present in the text. Let's apply BPE on this example text and observe the segmentation results."

segments = bpe_segment(text, num_iters=100)
print("初始文本:", text)
print("分词结果:", segments)
