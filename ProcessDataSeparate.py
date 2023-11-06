import pandas as pd
import fasttext

to_remove = [",", ".", "<", ">", "?", "/", ";", ":", "'", "!", "#", "$", "%", "^", "~",
             "*", "(", ")", "{", "}", "[", "]", "\\", "-", "_", "\n", "\t" "@", "&", "`"]

def camel_case_split(text):
    text_list = text.splitlines()
    res = ""
    for s in text_list:
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
        res = res + sentence +"\n"
    res = res.rstrip()
    return res

def filter(text):
  text = camel_case_split(text)
  for symbol in to_remove:
    text = text.replace(symbol, " ")
  while "  " in text:
    text = text.replace("  ", " ")
  return text


__all__ = ('get_docstrings', 'print_docstrings')

import ast

from itertools import groupby
from os.path import basename, splitext
import re
import tokenize, io

NODE_TYPES = {
    ast.ClassDef: 'Class',
    ast.FunctionDef: 'Function/Method',
    ast.Module: 'Module'
}


def get_docstrings(source):
    try:
        tree = ast.parse(source)
    except:
        return None

    for node in ast.walk(tree):
        if isinstance(node, tuple(NODE_TYPES)):
            try:
                docstring = ast.get_docstring(node)
            except:
                return None
            lineno = getattr(node, 'lineno', None)

            if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Str)):
                lineno = node.body[0].lineno - len(node.body[0].value.s.splitlines()) + 1

            yield (node, getattr(node, 'name', None), lineno, docstring)


def get_docstring(source, module='<string>'):
    ds = get_docstrings(source)
    if ds is None:
        return None
    docstrings = sorted(ds, key=lambda x: (NODE_TYPES.get(type(x[0])), x[1]))
    grouped = groupby(docstrings, key=lambda x: NODE_TYPES.get(type(x[0])))

    for type_, group in grouped:
        for node, name, lineno, docstring in group:
            name = name if name else module
            heading = "%s '%s', line %s" % (type_, name, lineno or '?')
            return docstring


def get_comments(code):
    buf = io.StringIO(code)
    res = ""
    for line in tokenize.generate_tokens(buf.readline):
        if line.type == tokenize.COMMENT:
            res += ((line.string).replace("#", "")) + "\n"
    return res


def split_code_docstring(code):
    docstring = get_docstring(code)
    if docstring is None:
        return None, code
    # docstring = docstring + "\n" + get_comments(code)
    docstring_lines = docstring.splitlines()
    code_lines = code.splitlines()
    code = ""
    for lne in code_lines:
        if lne not in docstring_lines:
            code = code + lne + "\n"
    while "  " in code:
        code = code.replace("  ", " ")
    return docstring, code


def trainLanguageModel(lang, embedding):
    w_filename1 = "Data/Texts/"+lang+"_text.txt"
    w_filename2 = "Data/Texts/"+lang+"_code.txt"
    f1 = open(w_filename1, "w+")
    f2 = open(w_filename2, "w+")
    types_of_data = ["train", "valid", "test"]
    print(lang)
    for tod in types_of_data:
        print(tod)
        filename = lang + "_" + tod + ".csv"
        data = pd.read_csv("Data/Texts/" + filename)
        targets = data["Target"].tolist()
        sources = data["Source"].tolist()
        targets = data["Target"].tolist()
        for s in sources:
            f1.write(filter(str(s)) + "\n")
        for t in targets:
            f2.write(filter(str(t)) + "\n")
    print("\n")
    f1.close()
    f2.close()
    model = fasttext.train_unsupervised(w_filename1, embedding, dim=300)
    model_path = "Data/Trained models/"+lang+"_text"
    model.save_model(model_path + ".bin")
    model = fasttext.train_unsupervised(w_filename2, embedding, dim=300)
    model_path = "Data/Trained models/"+lang+"_code"
    model.save_model(model_path + ".bin")

def generateEmbeddings(lang, embedding):
    model_path_text = "Data/Trained models/" + lang + "_text"
    model_path_code = "Data/Trained models/" + lang + "_code"
    ft_text = fasttext.load_model(model_path_text+".bin")
    ft_code = fasttext.load_model(model_path_code + ".bin")
    types_of_data = ["train", "valid", "test"]
    result = []
    filename = "Data/Texts/" + lang
    print(lang)
    for tod in types_of_data:
        print(lang, tod, filename)
        result = []
        data = pd.read_csv(filename + "_" + tod + ".csv")
        for index, row in data.iterrows():
            s = str(row["Source"])
            t = str(row["Target"])
            if s.isascii() == False or t.isascii() == False:
                continue
            s = filter(s)
            t = filter(t)
            s = ft_text.get_sentence_vector(s)
            t = ft_code.get_sentence_vector(t)
            result.append({"Source": t, "Target": s})
        df = pd.DataFrame(result)
        df_path = "Data/" + embedding + "/CodeSearch300/" + lang + "_" + tod + "_sep"
        df.to_csv(df_path + ".csv", index=False)


def processData(lang="python", embedding="cbow"):
    trainLanguageModel(lang, embedding)
    generateEmbeddings(lang, embedding)

embedding = "cbow"
processData("CodeSearch_PYTHON")
