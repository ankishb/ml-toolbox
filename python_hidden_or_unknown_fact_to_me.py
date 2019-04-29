 
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






import itertools

counter = itertools.count(start=5, step=5)
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))


data = [10, 20, 30, 40]

print(list(zip(itertools.count(), data)))
print(list(zip(range(10), data)))
print(list(itertools.zip_longest(range(10), data)))



counter = itertools.cycle([1,2,3])
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))

counter = itertools.cycle(('on','off'))
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))


squares = map(funcion, var1, var2, var3)
# these variables are taken by function

map(pow, range(10), itertools.repeat(2))
map(pow, [(0, 2), (1, 2), (2, 2)])
map(lambda x: x**2, range(10))

# to print these map output, use list(result)


letters = ['a', 'b', 'c']
result = itertools.combinations(letters,2)
for item in result:
    print(item)
result = itertools.permutations(letters,2)
# permutations: [(a,b), (b,a)] are 2 different possible outcomes, whereas in combinations [(a,b)] is possibles outcome.


numbers = [0,1,2,3]
result = itertools.product(numbers, repeat=4)
result = itertools.combinations_with_replacement(numbers, 4)





# sorting by abs of values
li = [-2, -4, 2, 5, -6]
sorted(li, key=abs)
sorted(li, key=lambda x: abs(x))

li.sort()# inplace sorting





from collections import namedtuple

Color = namedtuple('Color', ['red', 'green', 'blue'])
color = Color(55, 100, 255)
print(color[0])
print(color.red)

