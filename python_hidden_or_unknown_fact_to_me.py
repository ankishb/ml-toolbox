 Reference: https://codefellows.github.io/sea-f2-python-sept14/session05.html

 
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












Function arguments in variables

function arguments are really just

    a tuple (positional arguments)
    a dict (keyword arguments)

def f(x, y, w=0, h=0):
    print "position: %s, %s -- shape: %s, %s"%(x, y, w, h)

position = (3,4)
size = {'h': 10, 'w': 20}

>>> f( *position, **size)
position: 3, 4 -- shape: 20, 10








By "shallow copying" it means the content of the dictionary is not copied by value, but just creating a new reference.

In contrast, a deep copy will copy all contents by value.


b = a: Reference assignment, Make a and b points to the same object.

Illustration of 'a = b': 'a' and 'b' both point to '{1: L}', 'L' points to '[1, 2, 3]'.

b = a.copy(): Shallow copying, a and b will become two isolated objects, but their contents still share the same reference

Illustration of 'b = a.copy()': 'a' points to '{1: L}', 'b' points to '{1: M}', 'L' and 'M' both point to '[1, 2, 3]'.

b = copy.deepcopy(a): Deep copying, a and b's structure and content become completely isolated


Illustration of 'b = copy.deepcopy(a)': 'a' points to '{1: L}', 'L' points to '[1, 2, 3]'; 'b' points to '{1: M}', 'M' points to a different instance of '[1, 2, 3]'.

Ref: https://stackoverflow.com/questions/3975376/understanding-dict-copy-shallow-or-deep









What did * do?

It unpacked the values in list l as positional arguments. And then the unpacked values were passed to function ‘fun’ as positional arguments.

So, unpacking the values in list and changing it to positional arguments meant writing fun(*l) was equivalent to writing fun(1,2,3). Keep in mind that l=[1,2,3]

list_ = [1,2,3,4]
In [16]: fun(1, *list_)

def math_operation(operation, *args):
    if operation is 'sum':
        result = 0
        for arg in args:
            result += arg
    if operation is 'sub':
        result = 0
        for arg in args:
            result -= arg

list_ = [1,2,3,4,5,6]
math_operation('sum',list_)

We can’t write math_operation(args) because we need to unpack the values in the tuple args before operating further.


Let’s use ** from inside the function call. For this we want a dictionary. Remember, while using * in the function call, we required a list/tuple. For using ** in the function call, we require a dictionary.

def fun(a, b, c):
    print a, b, c

In [38]: dict_={'b':5, 'c':7}

Let’s call fun using ** in the function call.

In [39]: fun(1, **dict_)
1 5 7





