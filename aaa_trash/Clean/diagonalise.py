from itertools import permutations

def determinant(m: list) -> float:
    '''Calculates the determinant of a square matrix'''
    det = 0
    for p in permutations(range(len(m))):

        det += (1 - 2 * (sum([1 for i in range(len(m)) if i != p[i]]) % 2)) * \
               prod([m[i][p[i]] for i in range(len(m))])
    #if len(m) == 1:
    #    return m[0][0]
    #if len(m) == 2:
    #    return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    #det = 0
    #for i in range(len(m)):
    #    det += m[0][i] * ((-1) ** i) * determinant([row[:i] + row[i + 1:] for row in m[1:]])
    return det

def signature(p: list) -> int:
    '''Calculates the signature of a permutation'''
    return 