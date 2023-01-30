import pandas as pd
import fasttext

to_remove = [",", ".", "<", ">", "?", "/", ";", ":", "'", "!", "#", "$", "%", "^", "~",
             "*", "(", ")", "{", "}", "[", "]", "\\", "-", "_", "\t", "\n", "@", "&", "`"]

def camel_case_split(s):
    # Splits camel case names into multiple words
    idx = list(map(str.isupper, s))
    l = [0]
    for (i, (x, y)) in enumerate(zip(idx, idx[1:])):
        if x and not y:
            l.append(i)
        elif not x and y:
            l.append(i+1)
    l.append(len(s))
    words = [s[x:y] for x, y in zip(l, l[1:]) if x < y]
    sentence = ' '.join(words)
    while "  " in sentence:
      sentence = sentence.replace("  ", " ")
    return sentence

def filter(text):
    # Replaces camel case names with multiple words, and filters out irrelevant symbols
    text = camel_case_split(text)
    for symbol in to_remove:
        text = text.replace(symbol, " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def trainLanguageModel(lang, embedding, dimensions=300):
    types_of_data = ["train", "test", "valid"]
    print(lang)
    f = open("Data/Texts/CodeSearch_" + lang.upper() + "_text.txt", "w+")   # File containing list of text for model to learn from
    for tod in types_of_data:
        print(tod)
        filename = "CodeSearch_" + lang.upper() + "_" + tod + ".csv"
        data = pd.read_csv("Data/Texts/" + filename)
        sources = data["Source"].tolist()
        targets = data["Target"].tolist()
        for s in sources:
            f.write(filter(str(s)) + "\n")
        for t in targets:
            f.write(filter(str(t)) + "\n")
    print("\n")
    f.close()
    model = fasttext.train_unsupervised("Data/Texts/CodeSearch_" + lang.upper() + "_text.txt", embedding, dim=dimensions)  # Trains model for the passed embeddings and dimension sizes
    model.save_model("Data/Trained_models/" + lang.upper() + ".bin")

def generateEmbeddings(lang, embedding):
    ft = fasttext.load_model("Data/Trained_models/" + lang.upper() + ".bin")
    for tod in types_of_data:
        filename = "CodeSearch_" + lang.upper() + "_" + tod + ".csv"
        print(lang, tod, filename)
        data = pd.read_csv("Data/Texts/" + filename)
        result = []
        for index, row in data.iterrows():
            s = str(row["Source"])
            t = str(row["Target"])
            if s.isascii() == False or t.isascii() == False:    # Filters out non-ascii text
                continue
            s = ft.get_sentence_vector(filter(s))   # Get embeddings for the passed text
            t = ft.get_sentence_vector(filter(t))
            result.append({"Source": s, "Target": t})
        df = pd.DataFrame(result)
        df.to_csv("Data/"+embedding+"/CodeSearch300/" + filename, index=False)


def processData(lang="python", embedding="cbow", dimensions=300):
    trainLanguageModel(lang, embedding, dimensions=dimensions)
    generateEmbeddings(lang, embedding)

languages = ["python", "java", "go", "javascript", "ruby", "php"]
embedding = "cbow"  # Embedding model
dimensions = 300    # Embedded vectors' dimension size

for lang in languages:
    processData(lang=lang, embedding=embedding, dimensions=dimensions)