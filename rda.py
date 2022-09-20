import argparse
import socket
import time
import json
import numpy as np

from multiprocessing import Process, Value, Array

import mne


import utils

"""
Brain Vision Recorder Remote Data Access (RDA) Emulator
Ver 2.1 14th March, 2022
Ver 2.2 13th June, 2022
"""

NUMBER_OF_MARKER_POINTS = 1
MARKER_CHANNEL = 0
DEFAULT_CH_RESOLUTION = 0.1
NUMBER_OF_DATA_POINTS = 40
ENABLE_REALTIME = True
SLEEP_COEF = 0.88

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 51244  # Port to listen on (non-privileged ports are > 1023)

BV_RECORDER_ID = [-114, 69, 88, 67, -106, -55, -122, 76, -81, 74, -104, -69, -10, -55, 20, 80]
ids = np.array(BV_RECORDER_ID, dtype=np.int8)
BV_RECORDER_ID_BYTE = ids.tobytes()

#----------------------------------------------------------------------------------

def gen_header(id, msgsize, msgtype):

    header = bytes()

    #for m in range(0, 4):
    #    tmp = np.array(id[m]).astype(np.int32)
    #    header += tmp.tobytes()

    header += BV_RECORDER_ID_BYTE
    
    header += np.array(msgsize).astype(np.uint32).tobytes()
    header += np.array(msgtype).astype(np.uint32).tobytes()

    return header

def string2byte(string_array, format='ascii'):
    r = bytes()
    for string in string_array:
        r += bytes(string, format)
        r += np.array([0]).astype(np.int8).tobytes()
    return r

def gen_data_packets(eeg, markers, block, idx, data_points=NUMBER_OF_DATA_POINTS):

    if idx+data_points > eeg.shape[1]:

        if idx > eeg.shape[1]:
            pass
        else:
            print("Last Block")
            eeg_send = eeg[:, idx:]

            num_zeros = data_points - eeg_send.shape[1]
            zero_padding = np.zeros((eeg_send.shape[0], num_zeros))
            eeg_send = np.concatenate((eeg_send, zero_padding), axis=1)

            markers_send = markers[idx:]
            zero_padding = np.zeros(num_zeros)
            markers_send = np.concatenate((markers_send, zero_padding))
            markers_count = np.count_nonzero(markers_send)

            idx = -1 # Last block

    else:
        eeg_send = eeg[:, idx:idx+data_points]
        markers_send = markers[idx:idx+data_points]
        markers_count = np.count_nonzero(markers_send)
        idx += data_points

    block = block + 1

    n_row, n_column = eeg_send.shape

    eeg_send = np.reshape(eeg_send,n_row*n_column, order='F')

    body = bytes()

    tmp = np.array(block).astype(np.uint32)
    body += tmp.tobytes()
    tmp = np.array(data_points).astype(np.uint32)
    body += tmp.tobytes()
    tmp = np.array(markers_count).astype(np.uint32)
    body += tmp.tobytes()
    for sample_point in eeg_send:
        tmp = np.array(sample_point).astype(np.float32)
        body += tmp.tobytes()

    if markers_count > 0:
            tmp = np.argwhere(markers_send)
            markers_list_send = np.concatenate((tmp, markers_send[tmp]), axis=1)
            marker_block = bytes()
            for marker in markers_list_send:
                marker_block_single = bytes()
                position = marker[0]
                tmp = np.array(position).astype(np.uint32)
                marker_block_single += tmp.tobytes()

                points = NUMBER_OF_MARKER_POINTS
                tmp = np.array(points).astype(np.uint32)
                marker_block_single += tmp.tobytes()

                channel = MARKER_CHANNEL
                tmp = np.array(points).astype(np.int32)
                marker_block_single += tmp.tobytes()

                type_desc = string2byte(["S"+str(marker[1]), "S"+str(marker[1])])
                marker_block_single += type_desc

                marker_size = len(marker_block_single)+4
                tmp = np.array(marker_size).astype(np.uint32)

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

    data = np.zeros(n_ch*data_points)

    body = bytes()

    tmp = np.array(block).astype(np.uint32)
    body += tmp.tobytes()
    tmp = np.array(data_points).astype(np.uint32)
    body += tmp.tobytes()
    tmp = np.array([0]).astype(np.uint32)
    body += tmp.tobytes()
    for sample_point in data:
        tmp = np.array(sample_point).astype(np.float32)
        body += tmp.tobytes()

    msgtype = 4
    msgsize = len(body) + 24
    header = gen_header(id, msgsize, msgtype)
    data_send = header + body

    return data_send, block

def main(vhdr_fname, share):
    share[0] = -999
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

    markers = np.zeros((eeg.shape[1])).astype(np.uint8)
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

            body = bytes()

            tmp = np.array(raw.info['nchan']).astype(np.int32)
            body += tmp.tobytes()

            tmp = np.array(1000000/raw.info['sfreq']).astype(np.float64)
            body += tmp.tobytes()

            for m in range(raw.info['nchan']):
                tmp = np.array(resolution[m]).astype(np.float64)
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
            while not finish:
                data_send, block = gen_zero_packets(raw.info['nchan'], data_points, block)
                conn.sendall(data_send)
                if ENABLE_REALTIME:
                    time.sleep(NUMBER_OF_DATA_POINTS/raw.info['sfreq']*SLEEP_COEF)
                if share[0] == -999:
                    share[0] = 0
                #print(share[0])
                if share[0] == -100:
                    print("Started.")
                    idx = 0
                    streaming_finished = False
                    while not streaming_finished:
                        data_send, block, idx = gen_data_packets(eeg, markers, block, idx)
                        if idx == -1:
                            streaming_finished = True
                            share[0] = -99
                            print("Data Streaming Finished.")
                        conn.sendall(data_send)
                        if ENABLE_REALTIME:
                            time.sleep(NUMBER_OF_DATA_POINTS/raw.info['sfreq']*SLEEP_COEF)
                        

def interface(name, name_main_outlet='main', log_file=True, log_stdout=True, share=None, vhdr_fname=""):
    # ==============================================
    # This function is called from main module.
    # It opens LSL and communicate with main module.
    # ==============================================
    #
    # name : name of this module. main module will call this module with this name.
    #        This name will be given by main module.
    # name_main_outlet : name of main module's outlet. This module will find the main module with this name.
    #
    # share : instance of multiprocessing.Array, shared with parent module.
    # share[0] : -999 -> not ready

    share[0] = -999
    params = dict() # variable for receive parameters

    inlet = utils.getIntermoduleCommunicationInlet(name_main_outlet)
    print('LSL connected, %s' %name)

    p = Process(target=main, args=(vhdr_fname, share))
    p.start()

    while True:
        data, _ = inlet.pull_sample(timeout=0.01)
        if data is not None:
            if data[0].lower() == name:
                if data[1].lower() == 'cmd':
                    # ------------------------------------
                    # command
                    if data[2].lower() == 'start' or data[2].lower() == 'play':
                        share[0] = -100
                    elif data[2].lower() == 'stop':
                        print("receive stop command")
                        #stim_con.stop()
                    # modify here to add new commands
                    # ------------------------------------                        

                elif data[1].lower() == 'params':
                    # ------------------------------------
                    # parameters
                    params[data[2]] = json.loads(data[3])
                    #logger.debug('Param Received : %s' %str(params[data[2]]))
                    # ------------------------------------
                else:
                    raise ValueError("Unknown LSL data type received.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('f_dir')
    args = parser.parse_args()

    vhdr_fname = args.f_dir
    if ('.vhdr' in vhdr_fname) is False:
        vhdr_fname += ".vhdr"

    outlet = utils.createIntermoduleCommunicationOutlet('main', channel_count=4, id='rda_main')

    share = Array('i', [-999 for m in range(8)])
    #interface('rda', share=share)
    p = Process(target=interface, args=('rda', 'main', False, False, share, vhdr_fname))
    p.start()
    time.sleep(1)
    
    while True:
        while share[0] != 0:
            time.sleep(0.1)
        input("Press Any Key to Start")
        utils.send_cmd_LSL(outlet, 'rda', 'start')
        while share[0] != -99:
            time.sleep(0.1)
