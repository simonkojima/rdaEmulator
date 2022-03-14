import socket
import time
import numpy
import mne

"""
Brain Vision Recorder Remote Data Access (RDA) Emulator
Ver 2.1 14th March, 2022
"""

NUMBER_OF_MARKER_POINTS = 1
MARKER_CHANNEL = 0
DEFAULT_CH_RESOLUTION = 0.1
NUMBER_OF_DATA_POINTS = 40
ENABLE_REALTIME = True
SLEEP_COEF = 0.1

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 51244  # Port to listen on (non-privileged ports are > 1023)

vhdr_fname = "C:\\Users\\Simon\\Downloads\\bva_emulater\\bva_emulater\\20180514_P3000001.vhdr"

BV_RECORDER_ID = [-114, 69, 88, 67, -106, -55, -122, 76, -81, 74, -104, -69, -10, -55, 20, 80]
ids = numpy.array(BV_RECORDER_ID, dtype=numpy.int8)
BV_RECORDER_ID_BYTE = ids.tobytes()

#----------------------------------------------------------------------------------

def gen_header(id, msgsize, msgtype):

    header = bytes()

    #for m in range(0, 4):
    #    tmp = numpy.array(id[m]).astype(numpy.int32)
    #    header += tmp.tobytes()

    header += BV_RECORDER_ID_BYTE
    
    header += numpy.array(msgsize).astype(numpy.uint32).tobytes()
    header += numpy.array(msgtype).astype(numpy.uint32).tobytes()

    return header

def string2byte(string_array, format='ascii'):
    r = bytes()
    for string in string_array:
        r += bytes(string, format)
        r += numpy.array([0]).astype(numpy.int8).tobytes()
    return r

def gen_data_packets(eeg, markers, block, idx):
    
    if idx+data_points > eeg.shape[1]:

        if idx > eeg.shape[1]:
            pass
        else:
            print("Last Block")
            eeg_send = eeg[:, idx:]

            num_zeros = data_points - eeg_send.shape[1]
            zero_padding = numpy.zeros((eeg_send.shape[0], num_zeros))
            eeg_send = numpy.concatenate((eeg_send, zero_padding), axis=1)

            markers_send = markers[idx:]
            zero_padding = numpy.zeros(num_zeros)
            markers_send = numpy.concatenate((markers_send, zero_padding))
            markers_count = numpy.count_nonzero(markers_send)

            idx = -1 # Last block

    else:
        eeg_send = eeg[:, idx:idx+data_points]
        markers_send = markers[idx:idx+data_points]
        markers_count = numpy.count_nonzero(markers_send)
        idx += data_points

    block = block + 1

    n_row, n_column = eeg_send.shape

    eeg_send = numpy.reshape(eeg_send,n_row*n_column, order='F')

    body = bytes()

    tmp = numpy.array(block).astype(numpy.uint32)
    body += tmp.tobytes()
    tmp = numpy.array(data_points).astype(numpy.uint32)
    body += tmp.tobytes()
    tmp = numpy.array(markers_count).astype(numpy.uint32)
    body += tmp.tobytes()
    for sample_point in eeg_send:
        tmp = numpy.array(sample_point).astype(numpy.float32)
        body += tmp.tobytes()

    if markers_count > 0:
            tmp = numpy.argwhere(markers_send)
            markers_list_send = numpy.concatenate((tmp, markers_send[tmp]), axis=1)
            marker_block = bytes()
            for marker in markers_list_send:
                marker_block_single = bytes()
                position = marker[0]
                tmp = numpy.array(position).astype(numpy.uint32)
                marker_block_single += tmp.tobytes()

                points = NUMBER_OF_MARKER_POINTS
                tmp = numpy.array(points).astype(numpy.uint32)
                marker_block_single += tmp.tobytes()

                channel = MARKER_CHANNEL
                tmp = numpy.array(points).astype(numpy.int32)
                marker_block_single += tmp.tobytes()

                type_desc = string2byte(["S"+str(marker[1]), "S"+str(marker[1])])
                marker_block_single += type_desc

                marker_size = len(marker_block_single)+4
                tmp = numpy.array(marker_size).astype(numpy.uint32)

                marker_block_single = tmp.tobytes() + marker_block_single

                marker_block += marker_block_single

            body = body + marker_block

    msgtype = 4
    msgsize = len(body) + 24
    header = gen_header(id, msgsize, msgtype)
    data_send = header + body

    return data_send, block, idx

def gen_zero_packets(n_ch, data_points,block):
    block = block + 1

    data = numpy.zeros(n_ch*data_points)

    body = bytes()

    tmp = numpy.array(block).astype(numpy.uint32)
    body += tmp.tobytes()
    tmp = numpy.array(data_points).astype(numpy.uint32)
    body += tmp.tobytes()
    tmp = numpy.array([0]).astype(numpy.uint32)
    body += tmp.tobytes()
    for sample_point in data:
        tmp = numpy.array(sample_point).astype(numpy.float32)
        body += tmp.tobytes()

    msgtype = 4
    msgsize = len(body) + 24
    header = gen_header(id, msgsize, msgtype)
    data_send = header + body

    return data_send, block

#eog_ch_name = ["",""]


raw = mne.io.read_raw_brainvision(vhdr_fname)
data_points = NUMBER_OF_DATA_POINTS
resolution = list()
for m in range(raw.info['nchan']):
    resolution.append(DEFAULT_CH_RESOLUTION)

eeg = raw.get_data()*1000000 # convert unit from V to uV
eeg = eeg * 10 # somehow, streamed data from BV Recorder is multipled by 10.

events = mne.events_from_annotations(raw)
events = events[0]
events = events[1:]

markers = numpy.zeros((eeg.shape[1])).astype(numpy.int8)
for event in events:
    markers[event[0]] = event[2]

# initialize

block = -1
state = 0

finish = False

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")

        #---
        print("start")

        body = bytes()

        tmp = numpy.array(raw.info['nchan']).astype(numpy.int32)
        body += tmp.tobytes()

        tmp = numpy.array(1000000/raw.info['sfreq']).astype(numpy.float64)
        body += tmp.tobytes()

        for m in range(raw.info['nchan']):
            tmp = numpy.array(resolution[m]).astype(numpy.float64)
            body += tmp.tobytes()

        tmp = bytes()
        ch_names = raw.info['ch_names']
        body += string2byte(ch_names)

        msgtype = 1
        msgsize = len(body) + 24
        header = gen_header(id, msgsize, msgtype)

        data_send = header + body

        conn.sendall(data_send)
        if ENABLE_REALTIME:
            time.sleep(NUMBER_OF_DATA_POINTS/raw.info['sfreq']*SLEEP_COEF)

        #---
        # Data
        print("Press Ctrl+C to Start.")
        while not finish:
            try:
                data_send, block = gen_zero_packets(raw.info['nchan'], data_points, block)
                conn.sendall(data_send)
                if ENABLE_REALTIME:
                    time.sleep(NUMBER_OF_DATA_POINTS/raw.info['sfreq']*SLEEP_COEF)
            except KeyboardInterrupt:
                idx = 0
                streaming_finished = False
                while not streaming_finished:
                    data_send, block, idx = gen_data_packets(eeg, markers, block, idx)
                    if idx == -1:
                        streaming_finished = True
                        print("Data Streaming Finished.")
                        print("Press Ctrl+C to Start again.")
                    conn.sendall(data_send)
                    if ENABLE_REALTIME:
                        time.sleep(NUMBER_OF_DATA_POINTS/raw.info['sfreq']*SLEEP_COEF)
                    
        #---
"""
        msgtype = 3
        msgsize = 24
        header = gen_header(id, msgsize, msgtype)
        data_send = header
        state = 3
        #---
        conn.sendall(data_send)
        if ENABLE_REALTIME:
            time.sleep(NUMBER_OF_DATA_POINTS/raw.info['sfreq']*SLEEP_COEF)
        if state == 3:
            input("Press Enter to Terminate...")
            s.close()
            conn.close()
"""