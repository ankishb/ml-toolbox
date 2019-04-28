 
 Why python2 had unicode type, while python3 have str type.
 In Python 2 you have 8-bit str type and unicode unicode type. The difference is that in 8-bit string you can have only 256 different characters. If you use only ASCII characters, there is no problem; however if you want to use other characters (national, emoji , etc.) you need to encode them and you have many different encodings for that so the same str may have different meaning depending on the assumed encoding.



7. assert : This function is used for debugging purposes. Usually used to check the correctness of code. If a statement evaluated to true, nothing happens, but when it is false, “AssertionError” is raised . One can also print a message with the error, separated by a comma.




Future statements are special -- they change how your Python module is parsed, which is why they must be at the top of the file. They give new -- or different -- meaning to words or symbols in your file. From the docs:

    A future statement is a directive to the compiler that a particular module should be compiled using syntax or semantics that will be available in a specified future release of Python. The future statement is intended to ease migration to future versions of Python that introduce incompatible changes to the language. It allows use of the new features on a per-module basis before the release in which the feature becomes standard.

A propos print: print becomes a function in 3.x, losing its special property as a keyword. So it is the other way round.

>>> print

>>> from __future__ import print_function
>>> print
<built-in function print>
>>>




decorators: These are helpful when we want to use a function as an attributes.

class Employee:

    def __init__(self, first, last):
        self.first = first
        self.last = last

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last
    
    @fullname.deleter
    def fullname(self):
        print('Delete Name!')
        self.first = None
        self.last = None


emp_1 = Employee('John', 'Smith')
emp_1.fullname = "Corey Schafer"

print(emp_1.first)
print(emp_1.email)
print(emp_1.fullname)

del emp_1.fullname



