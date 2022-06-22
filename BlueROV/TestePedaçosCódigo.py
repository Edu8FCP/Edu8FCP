# MAIN FUNCTION

""" print('##############################################\n'
'#                                            #\n'
'#    Bem vindo ao programa de controlo!      #\n'
'#                                            #\n'
'##############################################\n')

while True: 

    ch = input()

    if ((ch == 'E') | (ch == 'e')):
        print("Saída")  

    if ((ch == 'C') | (ch == 'c')):
        print("Completo")

    if ((ch == 'S') | (ch == 's')):
        break

print("Fim do programa") 

    
 """
from argparse import ArgumentParser

parser = ArgumentParser(description=__doc__)
parser.add_argument("file",
    help="File that contains PingViewer sensor log file.")
args = parser.parse_args()

# Open log and begin processing
log = PingViewerLogReader(args.file)

for index, (timestamp, decoded_message) in enumerate(log.parser()):
    if index == 0:
        # Get header information from log
        # (parser has to do first yield before header info is available)
        print(log.header)

    # ask if processing
    yes = input("Continue and decode received messages? [Y/n]: ")
    if yes.lower() in ('n', 'no'):
        break

    print('timestamp:', repr(timestamp))
    print(decoded_message)
    # uncomment to confirm continuing after each message is printed
    #out = input('q to quit, enter to continue: ')
    #if out.lower() == 'q': break


### TESTE DE PARSE ###

#!/usr/bin/env python

#simplePingExample.py
# from ast import Str
# from locale import strcoll
# from brping import Ping1D
# import time
# import argparse

# from builtins import input

# ##Parse Command line options
# ############################

# parser = argparse.ArgumentParser(description="Ping python library example.")
# parser.add_argument('--device', action="store", required=False, type=str, help="Ping device port. E.g: /dev/ttyUSB0")
# parser.add_argument('--baudrate', action="store", type=int, default=115200, help="Ping device baudrate. E.g: 115200")
# parser.add_argument('--udp', action="store", required=False, type=str, default="192.168.2.2:9092", help="Ping UDP server. E.g: 192.168.2.2:9090")
# args = parser.parse_args()


# if args.device is None and args.udp is None:
#     parser.print_help()
#     exit(1)
# else: print("Parse Aceite")

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))