# Trie structure from the following article. Edited by Phillip Ngo
# Supports * wildcard. Example: adding "p*" means any word that starts with 'p' is a part of the Trie.
# https://towardsdatascience.com/implementing-a-trie-data-structure-in-python-in-less-than-100-lines-of-code-a877ea23c1a1

class LiwcTrieNode(object):
    def __init__(self, char: str):
        self.char = char
        self.children = []

def add(root, word: str):
    node = root
    for char in word:
        found_in_child = False
        for child in node.children:
            if child.char == '*':
                return
            if child.char == char:
                node = child
                found_in_child = True
                break
        if not found_in_child:
            new_node = LiwcTrieNode(char)
            node.children.append(new_node)
            node = new_node

def find(root, prefix: str):
    node = root
    if not root.children:
        return False
    for char in prefix:
        char_not_found = True
        for child in node.children:
            if child.char == '*':
                return True
            if child.char == char:
                char_not_found = False
                node = child
                break
        if char_not_found:
            return False
    return True
