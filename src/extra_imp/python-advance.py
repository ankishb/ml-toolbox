
Closure

There are a lot of usecases for something like this. It's especially useful for when you want to add or alter the 
functionality of your functions, without altering the code of the functions themselves. So for example, in this 
video I added logging capabilities to the add and subtract functions, all without adding any logging code to those 
functions themselves. This allows us to keep all of the logic separated into their own specific functions. If #you'd 
like to read more about it, I would suggest looking up "Aspect Oriented Programming". You will see a few more useful 
examples there. And that only touches on a couple of reasons why something like this would be useful. If #you've ever 
used a framework, like the Flask web framework for Python, then you'll notice that they have decorators all over their 
code. Those decorators are essentially closures that allow us to add routing and error handling capabilities to our 
functions. So you'll definitely find a usecase for something like this in the future. If you don't use it yourself, 
then you'll most likely run across someone else using it. So it's good to know.﻿







# Decorators
from functools import wraps


def my_logger(orig_func):
    import logging
    logging.basicConfig(filename='{}.log'.format(orig_func.__name__), level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper


def my_timer(orig_func):
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper

import time


@my_logger
@my_timer
def display_info(name, age):
    time.sleep(1)
    print('display_info ran with arguments ({}, {})'.format(name, age))

display_info('Tom', 22)





class decorator_class(object):

    def __init__(self, original_function):
        self.original_function = original_function

    def __call__(self, *args, **kwargs):
        print('call method before {}'.format(self.original_function.__name__))
        self.original_function(*args, **kwargs)


# Practical Examples

def my_logger(orig_func):
    import logging
    logging.basicConfig(filename='{}.log'.format(orig_func.__name__), level=logging.INFO)

    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper


def my_timer(orig_func):
    import time

    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper

import time