import itertools
import random

def getIntermoduleCommunicationInlet(target, timeout=None):
    from pylsl import StreamInlet, resolve_stream
    import time

    if timeout is None:
        timeout = float('inf')

    is_found = False
    start = time.time()
    while is_found is False:
        streams = resolve_stream('type', 'Markers')
        for stream in streams:
            if stream.name() == 'IntermoduleCommunication_' + target:
                inlet_intermodule = StreamInlet(stream)
                is_found = True
                return inlet_intermodule
        if time.time() - start > timeout:
            break

    if is_found is False:
        raise ValueError("Stream 'IntermoduleCommunication' was not found.")

def createIntermoduleCommunicationOutlet(name, type='Markers', channel_count=1, nominal_srate=0, channel_format='string', id=''):
    from pylsl import StreamInfo, StreamOutlet

    # StreamInfo(name, type, channel_count, nominal_srate, channel_format, source_id)
    # srate is set to IRREGULAR_RATE when nominal_srate = 0
    info = StreamInfo('IntermoduleCommunication_' + name, type, channel_count, nominal_srate, channel_format, id)
    outlet = StreamOutlet(info)

    return outlet

def send_cmd_LSL(outlet, target, cmd, params=None):
    if outlet.channel_count != 4:
        raise ValueError("channel_count of LSL outlet have to be 4.")
    
    import json
    params = json.dumps(params)
    outlet.push_sample([target, 'cmd', cmd, params])

def send_params_LSL(outlet, target, name_of_var, params=None):
    if outlet.channel_count != 4:
        raise ValueError("channel_count of LSL outlet have to be 4.")
    
    import json
    params = json.dumps(params)
    outlet.push_sample([target, 'params', name_of_var, params])

def _push_LSL(name, data, outlet):
    # not used
    if outlet.channel_count != 2:
        raise ValueError("channel_count of LSL outlet have to be 2.")

    import json
    outlet.push_sample("%s;"%name + json.dumps(data))

def _receive_LSL(val):
    # not used
    import json
    val = val.split(';')
    name = val[0]
    del(val[0])
    val = ';'.join(val)
    data = json.loads(val)

    return name, data
    
class Sudoku(object):

    """
    This source code for generating sudoku matrix is based on one from scripts of old framework which is based on matlab.
    Modified by Simon Kojima to adopt to Python version 3.x
    """

    def __init__(self):
        pass

    def _create_filter_for_vector(self, vector):
        return (lambda permutation: not self._has_any_same_elements_on_same_positions(permutation, vector))

    def _has_any_same_elements_on_same_positions(self, v1, v2):
        # TODO : impliment in more efficient way
        res = list()
        for idx, x in enumerate(list(v1)):
            res.append(x == v2[idx])
        return any(res)

    def generate_matrix(self, rows, columns):

        """
        This method creates "sudoku" matrix:
        1. Creates a list of all permutations of a vector
        2. Randomly selects an element.
        3. Filters a list of possible vectors to remove conflicting

        Note : should be rows <= columns
        """

        column_values = range(1, columns + 1)
        column_permutations = list(itertools.permutations(column_values))

        random.seed()
        
        matrix = [None] * rows
        for i in range(rows):
            column_permutation = random.choice(column_permutations)
            matrix[i] = list(column_permutation)
            column_permutations = list(filter(self._create_filter_for_vector(column_permutation), column_permutations))
        return matrix

def generate_stimulation_plan(n_stim_types = 5, itrs = 10):
    sudoku = Sudoku()
    mat = sudoku.generate_matrix(n_stim_types, n_stim_types)
    plan = mat[0]
    del mat[0]

    idx = -1
    for itr in range(1, itrs):
        is_finished = False
        while is_finished is False:
            idx += 1
            if idx >= len(mat):
                mat += sudoku.generate_matrix(n_stim_types, n_stim_types)
            is_finished = plan[-1] != mat[idx][0]
        plan += mat[idx]
        del mat[idx]
        idx = -1
        if len(mat) == 0:
            mat = sudoku.generate_matrix(n_stim_types, n_stim_types)
    return plan