import decode_sensor_binary
from argparse import ArgumentParser

# Parse arguments
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