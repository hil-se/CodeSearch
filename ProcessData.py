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
    w_filename = "Data/Texts/"+lang+".txt"
    f = open(w_filename, "w+")
    types_of_data = ["train", "valid", "test"]
    print(lang)
    for tod in types_of_data:
        print(tod)
        filename = lang + "_" + tod + ".csv"
        if lang=="DGMS":
            filename = "CodeSearch_PYTHON_" + tod + ".csv"
        data = pd.read_csv("Data/Texts/" + filename)
        targets = data["Target"].tolist()
        if lang=="DGMS":
            for t in targets:
                docstring, code = split_code_docstring(t)
                if code is None or docstring is None:
                    continue
                docstring_words = docstring.split()
                code_lines = code.splitlines()
                if len(docstring_words)<3 or len(code)<3:
                    continue
                f.write(filter(str(docstring)) + "\n")
                f.write(filter(str(t))+"\n")
        else:
            sources = data["Source"].tolist()
            targets = data["Target"].tolist()
            for s in sources:
                f.write(filter(str(s)) + "\n")
            for t in targets:
                f.write(filter(str(t)) + "\n")
    print("\n")
    f.close()
    model = fasttext.train_unsupervised(w_filename, embedding, dim=300)
    model_path = "Data/Trained models/"+lang
    model.save_model(model_path + ".bin")

def generateEmbeddings(lang, embedding):
    model_path = "Data/Trained models/" + lang
    ft = fasttext.load_model(model_path+".bin")
    types_of_data = ["train", "valid", "test"]
    result = []
    filename = "Data/Texts/" + lang
    for tod in types_of_data:
        print(lang, tod, filename)
        if lang=="DGMS":
            data = pd.read_csv("Data/Texts/CodeSearch_PYTHON_" + tod + ".csv")
            for index, row in data.iterrows():
                s = str(row["Source"])
                t = str(row["Target"])
                docstring, code = split_code_docstring(t)
                if code is None or docstring is None:
                    continue
                docstring_words = docstring.split()
                code_lines = code.splitlines()
                if len(docstring_words) < 3 or len(code) < 3:
                    continue
                s = ft.get_sentence_vector(filter(docstring))
                t = ft.get_sentence_vector(filter(code))
                result.append({"Source": s, "Target": t})
        else:
            result = []
            data = pd.read_csv(filename + "_" + tod + ".csv")
            for index, row in data.iterrows():
                s = str(row["Source"])
                t = str(row["Target"])
                if s.isascii() == False or t.isascii() == False:
                    continue
                s = filter(s)
                t = filter(t)
                s = ft.get_sentence_vector(s)
                t = ft.get_sentence_vector(t)
                result.append({"Source": t, "Target": s})
                df = pd.DataFrame(result)
                df_path = "Data/" + embedding + "/CodeSearch300/" + lang + "_" + tod
                df.to_csv(df_path + ".csv", index=False)
    if lang=="DGMS":
        df = pd.DataFrame(result)
        df.to_csv("Data/" + embedding + "/CodeSearch300/" + filename, index=False)


def processData(lang="python", embedding="cbow"):
    trainLanguageModel(lang, embedding)
    generateEmbeddings(lang, embedding)

embedding = "cbow"

processData("CodeSearch_PYTHON")
processData("AdvTest")
processData("DGMS")
