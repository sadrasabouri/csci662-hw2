import torch
import torch.nn as nn
import torch.nn.functional as F

# DO NOT MODIFY THIS FUNCTION
def setup_k_and_v():
    torch.manual_seed(447)

    key = torch.randn(4, 3)
    key /= torch.norm(key, dim=1, keepdim=True)
    key.round_(decimals=2)

    value = torch.randn(4, 3)
    value /= torch.norm(value, dim=1, keepdim=True)
    value.round_(decimals=2)

    # you can uncomment these if you want!
    # print(f'key:\n{key}')
    # print(f'value:\n{value}')
    return key, value

## THESE ARE WHERE YOUR ANSWERS GO
# We will call these functions during testing
def problem_1():
    key, value = setup_k_and_v()
    # TODO fill in the correct query vector to "select" the first value vector
    # can be a hardcoded value or an expression in terms of k and v
    S = 1000  # large scaling factor
    query_1 = key[0].unsqueeze(0) # to select the first item
    query_1 = S * query_1 # scale up to make the dot product more extreme

    # DO NOT REMOVE THIS
    return query_1

def problem_2():
    key, value = setup_k_and_v()
    # TODO fill in the query matrix which results in an identity mapping 
    # i.e. "select" all the value vectors
    # can be a hardcoded value or an expression in terms of k and v
    S = 1000  # large scaling factor
    query_2 = key.clone()  # same as problem 1 for for all the keys
    query_2 = S * query_2

    # DO NOT REMOVE THIS
    return query_2

def problem_3():
    key, value = setup_k_and_v()
    # TODO define a query vector which averages all the value vectors
    # can be a hardcoded value or an expression in terms of k and v

    # we should make a'i = 1/4 hence a = 0 this means that the query vector should be orthogonal to all key vectors
    # one cheat that can do this is to use null vector which has zero dot product with all vectors :)
    query_3 = torch.zeros_like(key)

    # here scaling would mess with the average vector so we don't scale

    # DO NOT REMOVE THIS
    return query_3

## TESTING FUNCTIONS PROVIDED FOR YOUR CONVENIENCE
# Everything below this line you can modify as you want
# We won't call these directly, but autograder tests will be very similar
# Strongly recommended to check your answers locally using these before you upload
def attention(query, key, value):
    """
    Note that we remove scaling for simplicity.
    """
    return F.scaled_dot_product_attention(query, key, value, scale=1)

def check_query(query, target, key, value):
    """
    Helper function for you to check if your query is close to the required target matrix.
    """
    a_out = attention(query, key, value)
    print(a_out)
    max_abs_diff = (target - a_out).abs().max()
    print("maximum absolute element-wise difference:", max_abs_diff)
    return max_abs_diff

def check_problem_1():
    """
    Helper function to check your solution to problem 1.
    """
    key, value = setup_k_and_v()
    query = problem_1()
    max_abs_diff = check_query(query, value[0], key, value)
    if max_abs_diff <= .05:
        print("problem 1 correct")
    else:
        print("problem 1 incorrect")

def check_problem_2():
    """
    Helper function to check your solution to problem 2.
    """
    key, value = setup_k_and_v()
    query = problem_2()
    max_abs_diff = check_query(query, value, key, value)
    if max_abs_diff <= .05:
        print("problem 2 correct")
    else:
        print("problem 2 incorrect")

def check_problem_3():
    """
    Helper function to check your solution to problem 3.
    """
    key, value = setup_k_and_v()
    query = problem_3()
    target = torch.reshape(value.mean(0, keepdims=True), (3,))
    max_abs_diff = check_query(query, target, key, value)
    if max_abs_diff <= .05:
        print("problem 3 correct")
    else:
        print("problem 3 incorrect")

def check_all_problems():
    check_problem_1()
    check_problem_2()
    check_problem_3()

# run this file as a script to check all problems!
if __name__ == "__main__":
    check_all_problems()
