from timeit import timeit
from csv import reader
import numpy as np

f = open('testing_values.csv', 'r')
rdr = reader(f)
header = next(rdr)
x = []
y = []

for row in rdr:
    x.append(float(row[0]))
    y.append(float(row[1]))

x = np.array(x)
y = np.array(y)

def opt_div(x,y):
    if x == 0:
        return 0
    else:
        return x/y

print(type(x[0]))
# print(y)

# setup_string = "x = 0.0;\
#                 y = 10.0"

# print(timeit("x/y", setup=setup_string, number=int(1e5), globals=globals()))
print(timeit("map(lambda a, b: a / b, x, y)", number=int(1e5), globals=globals()))
print(timeit("map(opt_div, x, y)", number=int(1e5), globals=globals()))
