import spacy
from nltk.tree import Tree



# Define a function to get all subpaths of a dependency tree
def get_token_subpaths(dep):
    subpaths = []
    if not list(dep.children):
        subpaths.append([dep])
    else:
        for child in dep.children:
            child_subpaths = get_token_subpaths(child)
            # subpaths += child_subpaths
            for child_subpath in child_subpaths:
                subpaths.append([dep] + child_subpath)
    return subpaths

def get_subpaths(doc):
    all_sub_paths = []
    for token in doc:
        subpaths = get_token_subpaths(token)
        all_sub_paths += subpaths

    all_sub_paths = [" - ".join([t.text for t in subpath]) for subpath in all_sub_paths]
    return all_sub_paths

def compute_tree_similarity(s1, s2, nlp=None):
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    paths1 = get_subpaths(doc1)
    paths2 = get_subpaths(doc2)

    common_tracks = 0

    for p1 in paths1:
        for p2 in paths2:
            if p1 == p2:
                common_tracks += 1

    other_1 = len(paths1) - common_tracks
    other_2 = len(paths2) - common_tracks

    # Jaccard similarity
    score = common_tracks/(common_tracks + other_1 + other_2)

    return score



if __name__ == '__main__':
    sent = "Sentence: Corticosteroid injections are commonly used as effective treatments for a variety of pain disorders ."
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sent)
    paths = get_subpaths(doc)
    for p in paths:
        print(p)