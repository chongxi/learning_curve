import time

# class Timeit(object):

# 	def __init__(self, func):
# 		self._func = func

# 	def __call__(self, *args, **kwargs):
# 		start_time = time.time()
# 		result = self._func(*args, **kwargs)
# 		print("elpsed time is {}".format(time.time()-start_time))
# 		return result

# 	def __get__(self, instance, owner):
# 		start_time = time.time()
# 		result = lambda *args, **kwargs: self._func(instance, *args, **kwargs)
# 		func = result
# 		print("elpsed time is {}".format(time.time()-start_time))
# 		return func

class bound_function_wrapper(object): 
    def __init__(self, wrapped):
        self.wrapped = wrapped 

    def __call__(self, *args, **kwargs):
    	start_time = time.time()
        result = self.wrapped(*args, **kwargs) 
        print("{0}::{1}, elpsed time is {2}".format(self.wrapped.im_class.__name__, 
        									       self.wrapped.im_func.__name__, 
        									       time.time()-start_time))
        return result


class Timeit(object): 
    def __init__(self, wrapped):
        self.wrapped = wrapped 

    def __get__(self, instance, owner):
        wrapped = self.wrapped.__get__(instance, owner)
        return bound_function_wrapper(wrapped) 

    def __call__(self, *args, **kwargs):
    	start_time = time.time()
        result = self.wrapped(*args, **kwargs) 
        print("{0}, elpsed time is {1}".format(self.wrapped.__name__, 
        	                                   time.time()-start_time))
        return result
