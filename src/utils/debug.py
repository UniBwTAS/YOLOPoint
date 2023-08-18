import numpy as np
# from utils.utils import toNumpy
from functools import wraps
import time

# def save_to_numpy(**kwargs):
#     # used for debugging
#     input_dict = {}
#     print(kwargs)
#     for key in kwargs:
#         try:
#             input_dict.update({key: toNumpy(kwargs[key])})
#         except:
#             input_dict.update({key: kwargs[key]})
#     np.savez_compressed('junk/test_input', **input_dict)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

# save_to_numpy(descriptors=descriptors, descriptors_warped=descriptors_warped, homographies=homographies, mask_valid=mask_valid,
#               cell_size=cell_size, lambda_d=lambda_d, margin_pos=margin_pos, margin_neg=margin_neg, device=device)