#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trie.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

from string import ascii_letters


class TrieNode:
    def __init__(self,):
        self.isWord = False
        self.s = {c: None for c in ascii_letters}  # children candidate


def add(T, w, i=0):
    """
    :param T: trie
    :param string w: word to be added to T
    :returns: new trie consisting of w added into T
    :complexity: O(len(w))
    """
    if T is None:
        T = TrieNode()
    if i == len(w):   # 叶子节点
        T.isWord = True
    else:
        T.s[w[i]] = add(T.s[w[i]], w, i+1)
    return T


def Trie(word_set):
    """
    :param word_set: set of words
    :returns: trie containing all words from S
    :complexity: linear in total word sizes from S
    """
    T = None
    for w in word_set:
        T = add(T, w)
    return T


def spell_check(trie, w):
    """
    :param T: trie encoding the dictionary
    :param w: given word
    :returns: a closest word from the dictionary
    :complexity: linear if distance was constant
    """
    assert trie is not None
    dist = 0
    while True:
        u = search(trie, dist, w)
        if u is not None:
            return u
        dist += 1


def search(trie, dist, w, i=0):
    """搜索与w[i:]的edit distance最大为dist的word"""
    if i == len(w):
        if trie is not None and trie.isWord and dist == 0:
            return ""
        else:
            return None
    if trie is None:
        return None

    f = search(trie.s[w[i]], dist, w, i + 1)       # matching
    if f is not None:
        return w[i] + f
    if dist == 0:
        return None
    for c in ascii_letters:
        f = search(trie.s[c], dist - 1, w, i)      # insertion
        if f is not None:
            return c + f
        f = search(trie.s[c], dist - 1, w, i + 1)  # substitution
        if f is not None:
            return c + f
    return search(trie, dist - 1, w, i + 1)        # deletion



import unittest

class TestTryalgo(unittest.TestCase):
    def test_trie(self):
        T = Trie(["as", "porc", "pore", "pre", "pres", "pret"])
        for w, closest in zip(["a", "aas", "ass", "pars", "por", "pes", "pred", "pire", "brzlgrmpf"],
                              ["as", "as", "as", "porc", "porc", "pres", "pres", "pore", "pres"]):
            self.assertEqual(spell_check(T, w), closest)


if __name__ == '__main__':
    # unittest.main()

    T = Trie(["as", "porc", "pore", "pre", "pres", "pret"])
    spell_check(T, "aas")
