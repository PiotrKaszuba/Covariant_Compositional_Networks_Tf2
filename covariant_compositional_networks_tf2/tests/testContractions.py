from string import ascii_lowercase
from itertools import combinations


def getContractions(k, k_increase, feature_vector_shape, channels_in=None):
    ascii_chi = ascii_lowercase[:9]
    ascii_f = ascii_lowercase[9:18]
    # max feature vector dimensions = 7
    ascii_feature_vector = ascii_lowercase[18:25]
    ascii_channels_in = ascii_lowercase[25]
    ascii_contractions = ascii_f

    str_to_join = []
    str_feature_vector = []
    if channels_in is not None:
        str_to_join.append(ascii_channels_in)  # channels in
    for i in range(k + k_increase):
        str_to_join.append(ascii_contractions[i])  # all dims to contract
    for i in range(len(feature_vector_shape)):
        str_to_join.append(ascii_feature_vector[i])  # feature vectors
        str_feature_vector.append(ascii_feature_vector[i])
    str_to_join.append('->')
    str_to_join.append(ascii_channels_in)  # channels_in
    contract_symbols = ascii_contractions[:k + k_increase]
    combinations_of_out_symbols = combinations(contract_symbols, k)
    expressions = [''.join(str_to_join) + ''.join(combination) + ''.join(str_feature_vector) for combination in
                   combinations_of_out_symbols]
    # expressions = [''.join(str_to_join) + ''.join(combination) + ''.join(str_feature_vector)]



    return expressions

print(getContractions(2,3, [1], [3]))