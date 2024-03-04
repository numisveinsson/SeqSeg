
def get_smoothing_params(radius, scale_unit, mega_sub = False):
    num_iterations = 3
    if not mega_sub:
        if radius > 1 * scale_unit: num_iterations = 12
        elif radius > 0.5 * scale_unit:
            print("Small radius; less smoothing")
            num_iterations = 6
    else:
        # less smoothing for bigger volumes
        if radius > 1 * scale_unit: num_iterations = 6
        elif radius > 0.5 * scale_unit:
            print("Small radius; less smoothing")
            num_iterations = 5

    return num_iterations