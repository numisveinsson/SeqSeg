
def get_smoothing_params(radius, scale_unit, mega_sub = False):
    
    if not mega_sub:
        num_iterations = 6
        if radius > 1 * scale_unit: num_iterations = 24
        elif radius > 0.5 * scale_unit:
            print("Small radius; less smoothing")
            num_iterations = 16
    else:
        num_iterations = 3
        # less smoothing for bigger volumes
        if radius > 1 * scale_unit: num_iterations = 7
        elif radius > 0.5 * scale_unit:
            print("Small radius; less smoothing")
            num_iterations = 5

    return num_iterations

class SkipThisStepError(Exception):
    pass