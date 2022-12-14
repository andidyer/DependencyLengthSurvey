from conllu import TokenList, TokenTree


def get_tree_weight(tree: TokenTree):
    """"Recursive function for getting the total number of dependents in a tree"""
    weight = 0
    for branch in tree.children:
        weight += get_tree_weight(branch)
    weight += 1

    return weight


def make_index_mapping(tokenlist: TokenList) -> dict:
    index_mapping = {0: 0, None: None}
    i = 1
    for token in tokenlist:
        if isinstance(token["id"], int):
            index_mapping[token["id"]] = i
            i += 1
    return index_mapping


def fix_tree_indices(tokenlist: TokenList):
    new_tokenlist = tokenlist.copy()

    index_mapping = make_index_mapping(new_tokenlist)
    for i, token in enumerate(new_tokenlist):
        token_id = token["id"]
        token_head = token["head"]
        new_tokenlist[i]["id"] = index_mapping[token_id]
        new_tokenlist[i]["head"] = index_mapping[token_head]

    return new_tokenlist
