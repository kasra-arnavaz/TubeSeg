import time
from cached_property import cached_property
class A:

    def a(self, i, f):
        return [x for x in range(i, f)]


    def b(self):
        x = sum(self.a(0, 50000000))
        return x + x

    @property
    def p(self):
        return sum(self.a(0, 50000000))

    def c(self):
        return self.p + self.p

if __name__ == '__main__':

    time_1 = time.time()
    A().b()
    time_2 = time.time()
    A().c()
    time_3 = time.time()
    print(time_2 - time_1)
    print(time_3 - time_2)

        

