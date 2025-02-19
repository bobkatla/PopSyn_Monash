"""
Contain variables that are shared across the SAA operations

Most likely to be accessed and updated multiple times
Careful with data corruption
"""

# Zero cells
zero_cells = {}

def update_zero_cells(att, value):
    if att not in zero_cells:
        zero_cells[att] = {value}
    else:
        zero_cells[att].add(value)

def get_zero_cells_all():
    return zero_cells

def get_zero_cells_att(att):
    return zero_cells[att]




