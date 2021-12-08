from parent import *

class Student(Person):
    def __init__(self):
        super().__init__()
        self.b = 2


me = Student()
me.c = 3

print(me.a, me.b, me.c)