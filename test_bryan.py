import numpy as np
import vtk

def mult_by_3(vector):

    return 3*vector

if __name__=='__main__':

    a = np.array([1,2,3])
    b = 3*a
    c = mult_by_3(a)

    print('I solved b: ', b)
    
    import pdb; pdb.set_trace()

    print('I solved c: ', d)
