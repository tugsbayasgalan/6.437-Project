import numpy as np
import math
import random as rand
from collections import defaultdict

data_path = "data/"
state_matrix = np.genfromtxt(data_path + "letter_transition_matrix.csv", delimiter=",")

state_matrix[state_matrix == 0] = 1e-20
log_state_matrix = np.log(state_matrix)


letters = 'abcdefghijklmnopqrstuvwxyz .'

letter_to_num = {}
for index, char in enumerate(letters):
    letter_to_num[char] = index

num_to_letter = {}
for index, char in enumerate(letters):
    num_to_letter[index] = char

def build_cipher_matrix(f, ciphertext):

    message = convert_to_plain(f, ciphertext)

    N = 28
    def fixed_array():
        return [0]*N
    freq_dict = defaultdict(fixed_array)
    total_len = len(message)
    for index, char in enumerate(message):
        if index > 0:
            prev = message[index-1]
            prev_index = letter_to_num[prev]
            freq_dict[char][prev_index] += 1

    matrix = []
    for char in letters:
        char_array = [float(val) for val in freq_dict[char]]
        matrix.append(char_array)

    np_matrix = np.matrix(matrix)
    return np_matrix

def random_neighbor(cipher_matrix, f):
    rand_ind1 = np.random.randint(0, 26)
    rand_ind2 = np.random.randint(0, 25)
    if rand_ind2 >= rand_ind1:
        rand_ind2 += 1

    f_list = list(f)
    f_list[rand_ind1], f_list[rand_ind2] = f[rand_ind2], f[rand_ind1]
    f_p = "".join(f_list)
    return swap(cipher_matrix, rand_ind1, rand_ind2), f_p


def swap(cipher_matrix, a, b):

    cipher_matrix_v2 = np.copy(cipher_matrix)

    cipher_matrix_v2[:,[a, b]] = cipher_matrix_v2[:,[b, a]]
    cipher_matrix_v2[[a,b]] = cipher_matrix_v2[[b,a]]

    return cipher_matrix_v2


def compute_log_prob(log_state_matrix, observed_matrix):


    mult_matrix = log_state_matrix * observed_matrix
    result = np.sum(mult_matrix)
    return result


def swap_letter(f):
    rand_ind1 = np.random.randint(0, 26)
    rand_ind2 = np.random.randint(0, 25)
    if rand_ind2 >= rand_ind1:
        rand_ind2 += 1

    f_list = list(f)
    f_list[rand_ind1], f_list[rand_ind2] = f[rand_ind2], f[rand_ind1]
    f_p = "".join(f_list)

    return f_p


def trim(val, min, max):
    if val < min: return min
    if val > max: return max
    return val

def find_period_whitespace(ciphertext):

    def set_int():
        return [set(), 0]

    frequency_dict = defaultdict(set_int)

    total_len = len(ciphertext)

    for index, char in enumerate(ciphertext):
        if index < total_len - 1:
            next = ciphertext[index + 1]
            frequency_dict[char][0].add(next)
            frequency_dict[char][1] += 1

    max_value = 0
    period = ""
    white = ""
    for key in frequency_dict:
        if len(frequency_dict[key][0]) == 1:

            if frequency_dict[key][1] > max_value:
                max_value = frequency_dict[key][1]
                period = key
                white = frequency_dict[key][0].pop()

    return period, white

def random_initialize(f, cipher_period, cipher_white):

    period_index = letter_to_num[cipher_period]
    white_index = letter_to_num[cipher_white]

    list_f = list(f)

    list_f[period_index], list_f[27] = list_f[27], list_f[period_index]
    list_f[white_index], list_f[26] = list_f[26], list_f[white_index]

    f_prime = "".join(list_f)

    rand_min = 20
    rand_max = 25

    rand_int = np.random.randint(rand_min, rand_max)
    count = 0
    while count < rand_int:
        f_prime  = swap_letter(f_prime)
        count += 1
    return f_prime


def decode(ciphertext, output_file_name):
    #some preprocessing
    new_line_indices = set()

    for index, char in enumerate(ciphertext):
        if char == '\n':
            new_line_indices.add(index)

    formatted_ciphertext  = ciphertext.replace("\n", "")
    cipher_period, cipher_white = find_period_whitespace(formatted_ciphertext)
    end = cipher_white + cipher_period


    #hyperparameters
    num_trial = 10
    num_epoch = 20000
    mix_time = 10000
    max_non_accepted = 50000


    #MCMC
    different_functions = []
    trial = 0

    while trial < num_trial:

        print "Trial Number: ", trial + 1
        f = random_initialize(letters, cipher_period, cipher_white)
        cipher_matrix = build_cipher_matrix(f, formatted_ciphertext)

        i = 1
        count = 1
        distribution = defaultdict(int)

        rejected = 0

        while count < num_epoch:

            f_given_y = compute_log_prob(log_state_matrix, cipher_matrix)
            cipher_matrix_cand, f_prime = random_neighbor(cipher_matrix, f)
            f_given_y_prime = compute_log_prob(log_state_matrix, cipher_matrix_cand)

            ratio = trim(f_given_y_prime - f_given_y, -50, 0)
            a = min(1, math.exp(ratio))

            u = rand.random()
            if u < a:
                cipher_matrix = cipher_matrix_cand
                f = f_prime
            else:
                rejected += 1

            #if too many samples are rejected
            if rejected >= max_non_accepted:
                print "This trial is rejected"
                break

            i += 1
            if i >= mix_time:
                distribution[f] += 1
                count += 1
            if count % 2500 == 0:
                print "Currently at: ", count

        #choose highest probability function
        max_value = max([(v, k) for k, v in distribution.iteritems()])
        cipher_func = max_value[1]
        different_functions.append(cipher_func)
        trial += 1



    #choose most occuring ciphertext
    count_cipher = defaultdict(int)
    for cipher in different_functions:
        count_cipher[cipher] += 1

    cipher_func = max([(v,k) for k, v in count_cipher.iteritems()])[1]

    message = convert_to_plain(cipher_func, formatted_ciphertext)

    formatted_message = ""

    for index, char in enumerate(message):
        if index in new_line_indices:
            formatted_message += '\n'
        formatted_message += char


    f = open(output_file_name, "w")
    f.write(formatted_message)
    f.close()
    return cipher_func




#helper functions for testing purposes
def generate_random_cipher_func():

    l = list(letters)
    rand.shuffle(l)
    result = ''.join(l)
    return result

def convert_to_plain(cipher_func, ciphertext):
    result = ""

    convert_dict = {}

    for index, char in enumerate(cipher_func):
        convert_dict[char] = num_to_letter[index]

    for char in ciphertext:
        result += convert_dict[char]

    return result
def generate_cipher_text(plain_text):

    cipher_func = generate_random_cipher_func()

    convert_dict = {}

    for index, char in enumerate(cipher_func):
        convert_dict[num_to_letter[index]] = char


    output = ""
    for char in plain_text:
        output += convert_dict[char]

    return output

def calculate_accuracy(cipher_func, cipher_text, plain_text):

    decoded = convert_to_plain(cipher_func, cipher_text)

    assert(len(decoded) == len(plain_text))

    mismatch = 0
    for i in range(len(decoded)):
        if decoded[i] != plain_text[i]:
            mismatch += 1

    print "Missed " + str(mismatch) + " out of " + str(len(decoded))


#testing
if __name__ == '__main__':

    with open('data/plaintext_warandpeace.txt', 'r') as myfile:
        test_plain= myfile.read().replace("\n", "")

    cipher_test = generate_cipher_text(test_plain)
    cipher_func = decode(cipher_test, "stuff.txt")
    print "Cipher Func", cipher_func
    calculate_accuracy(cipher_func, cipher_test, test_plain)
