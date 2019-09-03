import numpy as np
from utils import *


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    distance = 0.0

    dot = np.dot(u, v)

    norm_u = np.sqrt(np.sum(np.square(u)))
    norm_v = np.sqrt(np.sum(np.square(v)))

    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    words = word_to_vec_map.keys()
    max_cosine_sim = -100  # Initialize max_cosine_sim to a large negative number
    best_word = None       # Initialize best_word with None, it will help keep track of the word to output

    for w in words:
        if w in [word_a, word_b, word_c]:
            continue

        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word


def gender_bias_axis(wordpairs, word_to_vec_map):
    """find bias axis

    wordpairs -- list of (word1, word2) that has gender information.
                 eg. 'man' and 'woman', 'boy' and 'girl', 'father' and 'mother'...
    """
    g = None
    for (word1, word2) in wordpairs:
        g += word_to_vec_map[word1] - word_to_vec_map[word2]  # 注意保持男性词和女性词在wordpairs中的顺序一致
    g = g / len(wordpairs)
    return g


def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    e = word_to_vec_map[word]

    # bias轴上的投影向量
    e_biascomponents = np.dot(e, g) / np.linalg.norm(g)**2 * g

    # Neutralize, 转换为垂直于bias轴的vector
    e_debiased = e - e_biascomponents

    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    mu = (e_w1 + e_w2) / 2
    # 均值在bias轴的投影vector
    mu_B = np.dot(mu, bias_axis) / np.dot(bias_axis,bias_axis) * bias_axis
    mu_orth = mu - mu_B  # 垂直bias轴的均值vector

    # w1, w2在bias轴的投影
    e_w1B = np.dot(e_w1, bias_axis)/np.dot(bias_axis,bias_axis) * bias_axis
    e_w2B = np.dot(e_w2, bias_axis)/np.dot(bias_axis,bias_axis) * bias_axis

    # 在bias轴上，使w1, w2对称化
    corrected_e_w1B = np.sqrt(np.abs(1-np.dot(mu_orth,mu_orth))) / np.linalg.norm(e_w1B - mu_B) * (e_w1B - mu_B)
    corrected_e_w2B = np.sqrt(np.abs(1-np.dot(mu_orth,mu_orth))) / np.linalg.norm(e_w2B - mu_B) * (e_w2B - mu_B)

    # 对称化的w1, w2加上 相同的orthogonal轴vector
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2


if __name__ == '__main__':
    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    gender_pairs = [('man', 'woman'), ('boy', 'girl'), ('father', 'mother'), ('grandfather', 'grandmother')]

    g = gender_bias_axis(gender_pairs, word_to_vec_map)

    # 示例
    print('>>> biased words and their similarities:')
    word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
                'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
    for w in word_list:
        print (w, cosine_similarity(word_to_vec_map[w], g))

    print('>>> debias gender by neutralize:')
    e = "receptionist"
    print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

    e_debiased = neutralize("receptionist", g, word_to_vec_map)
    print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))

    # debias word vector
    print('>>> debias word vector:')
    print("cosine similarities before equalizing:")
    print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
    print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
    print()
    # ("man", "woman")可以是任何应该在gender轴上对称的word pairs，可以通过分类算法求解
    e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
    print("cosine similarities after equalizing:")
    print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
    print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))