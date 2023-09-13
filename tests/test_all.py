#from contextlib import contextmanager
#import sys, os
#import unittest

#from test_data_handler import *
#from versatile import *

#@contextmanager
#def suppress_stdout():
#    with open(os.devnull, "w") as devnull:
#        old_stdout = sys.stdout
#        sys.stdout = devnull
#        try:  
#            yield
#        finally:
#            sys.stdout = old_stdout

#if __name__ == '__main__':
#    with suppress_stdout():
#        unittest.main()