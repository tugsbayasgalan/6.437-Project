import numpy as np
import math
import random as rand
import string
from collections import defaultdict

data_path = "project_part_I/"
letter_probs = np.genfromtxt(data_path + "letter_probabilities.csv", delimiter=",").tolist()
state_matrix = np.genfromtxt(data_path + "letter_transition_matrix.csv", delimiter=",")



log_state_matrix = np.log(state_matrix)




letters = 'abcdefghijklmnopqrstuvwxyz .'

letter_to_num = {}
for index, char in enumerate(letters):
    letter_to_num[char] = index

num_to_letter = {}
for index, char in enumerate(letters):
    num_to_letter[index] = char

def build_cipher_matrix(ciphertext):

    N = 28
    def fixed_array():
        return [0]*N
    freq_dict = defaultdict(fixed_array)
    total_len = len(ciphertext)
    for index, char in enumerate(ciphertext):

        if index < total_len - 1:
            next = ciphertext[index+1]
            index_next = letter_to_num[next]
            freq_dict[char][index_next] += 1


    matrix = []
    for char in letters:
        char_array = [float(val) for val in freq_dict[char]]
        char_sum = sum(char_array)
        char_prob = [val/char_sum for val in char_array]

        matrix.append(char_prob)

    np_matrix = np.matrix(matrix)
    return np_matrix

def swap(cipher_matrix, a, b):


    cipher_matrix[:,[a, b]] = cipher_matrix[:,[b, a]]
    cipher_matrix[[a,b]] = cipher_matrix[[b,a]]

    return cipher_matrix





def compute_log_prob(y_number, inverse_function, target= -float('inf')):

    current = letter_to_num[inverse_function[y_number[0]]]

    log_prob = math.log(letter_probs[current])
    for number in y_number[1:]:
        previous = current
        current = letter_to_num[inverse_function[number]]
        if log_state_matrix[current, previous] == -float("inf"):
            return -float('inf')
        if log_prob - target < -10:
            return log_prob
        log_prob += log_state_matrix[current, previous]

    return log_prob


def swap_letter(f, inv_f):
    rand_ind1 = np.random.randint(0, 28)
    rand_ind2 = np.random.randint(0, 27)
    if rand_ind2 >= rand_ind1:
        rand_ind2 += 1

    f_list = list(f)
    f_list[rand_ind1], f_list[rand_ind2] = f[rand_ind2], f[rand_ind1]
    f_p = "".join(f_list)

    inv_f_list = list(inv_f)
    inv_f_list[letter_to_num[f[rand_ind1]]], inv_f_list[letter_to_num[f[rand_ind2]]] = inv_f[letter_to_num[f[rand_ind2]]], inv_f[letter_to_num[f[rand_ind1]]]
    inv_f_p = "".join(inv_f_list)

    return f_p, inv_f_p


def trim(val, min, max):
    if val < min: return min
    if val > max: return max
    return val

def find_period_whitespace(ciphertext):

    frequency_dict = defaultdict(set)

    total_len = len(ciphertext)

    for index, char in enumerate(ciphertext):
        if index < total_len - 1:
            next = ciphertext[index + 1]
            frequency_dict[char].add(next)

    for key in frequency_dict:
        if len(frequency_dict[key]) == 1:
            print key
            period = key
            whitespace = frequency_dict[key].pop()
            print whitespace
            return period, whitespace
    return False

def is_valid_cipher_func(cipher_func, cipher_period, cipher_white):

    return cipher_func[-2:] == cipher_white + cipher_period



def randomize_input(f, inverse_f):

    rand_min = 3
    rand_max = 7

    rand_int = np.random.randint(rand_min, rand_max)
    count = 0
    while count < rand_int:
        f, inverse_f = swap_letter(f, inverse_f)
        count += 1
    return f, inverse_f


def decode(ciphertext, output_file_name):
    #some preprocessing

    cipher_period, cipher_white = find_period_whitespace(ciphertext)
    print "Ending:", cipher_white + cipher_period
    num_epoch = 10000
    mix_time = 500
    max_non_accepted = 2000

    #convert ciphertext to array of numbers
    cipher_num = []
    for char in ciphertext:
        cipher_num.append(letter_to_num[char])


    #MCMC

    all_results = []
    trial = 0

    while trial < 10:
        f, inverse_f = randomize_input(letters, letters)
        i = 1
        count = 1
        distribution = defaultdict(int)
        f_given_y = compute_log_prob(cipher_num, inverse_f)
        non_accepted = 0
        while count < num_epoch:

            f_prime, inverse_f_prime = swap_letter(f, inverse_f)
            f_given_y_prime = compute_log_prob(cipher_num, inverse_f_prime, f_given_y)

            if f_given_y_prime == -float('inf'):
                a = 0

            if f_given_y == -float('inf'):
                a = 1
            else:
                ratio = trim(f_given_y_prime - f_given_y, -50, 0)
                a = min(1, math.exp(ratio))

            u = rand.random()
            if u < a:
                f = f_prime
                f_given_y = f_given_y_prime
                inverse_f = inverse_f_prime

            else:
                non_accepted += 1


            #if samples started getting rejected a lot
            if non_accepted >= max_non_accepted:
                break


            i += 1
            if i >= mix_time:
                distribution[f] += 1
                count += 1
            if count % 2000 == 0:
                print "Currently at: ", count

        max_value = max([(v, k) for k, v in distribution.iteritems()])
        print "Current cipher func: ", max_value
        cipher_func = max_value[1]

        if is_valid_cipher_func(cipher_func, cipher_period, cipher_white):
            print "Potential Candidate Found"
            all_results.append(cipher_func)
            trial += 1



    count_cipher = defaultdict(int)
    for cipher in all_results:
        count_cipher[cipher] += 1

    cipher_func = max([(v,k) for k, v in count_cipher.iteritems()])[1]

    message = convert_to_plain(cipher_func, ciphertext)

    f = open(output_file_name, "w")
    f.write(message)
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
    print cipher_func

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
    with open('project_part_II/plaintext_paradiselost.txt', 'r') as myfile:
        test_plain= myfile.read().replace('\n', '')
    print len(test_plain)
    cipher_test = generate_cipher_text(test_plain)
    #cipher_matrix = build_cipher_matrix(cipher_test)

    #cipher_matrix = np.matrix([[0,1,0,0], [0, 0, 1, 0], [0,0,0,1],[0,0,0,0]])
    #stuff = swap(cipher_matrix, 0 , 1)
    #print stuff

    cipher_func = decode(cipher_test, "stuff.txt")
    print "Cipher Func", cipher_func

    calculate_accuracy(cipher_func, cipher_test, test_plain)
