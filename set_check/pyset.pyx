from libcpp.set cimport set
from cython.operator cimport dereference as deref, preincrement as inc



def make_set(int k):
    cdef set[int] cpp_set
    
    for i in range(k): #Iterate through the set as a c++ set
        cpp_set.insert(i)

    for i in cpp_set: #Iterate through the set as a c++ set
        print i

    #Iterate through the set using c++ iterators.
    cdef set[int].iterator it = cpp_set.begin()
    while it != cpp_set.end():
        print deref(it)
        inc(it)

    return cpp_set    #Automatically convert the c++ set into a python set