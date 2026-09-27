import sys
import argparse

from recording.summary import print_summary
from recording.inspect import print_scan_stats, print_dump


def main() -> None:
    argv = sys.argv[1:]
    # Bare `python -m recording run.h5` keeps working as a summary.
    if argv and argv[0] not in ('summary', 'scans', 'dump', '-h', '--help'):
        argv = ['summary'] + argv

    parser = argparse.ArgumentParser(prog = 'recording',
                                     description = 'Inspect a signal recording')
    sub = parser.add_subparsers(dest = 'command', required = True)

    p_summary = sub.add_parser('summary', help = 'Per-stream counts, rates and sizes')
    p_summary.add_argument('path', help = 'Path to a .h5 recording file')

    p_scans = sub.add_parser('scans', help = 'Per-scan dropout statistics (empty LiDAR sectors)')
    p_scans.add_argument('path', help = 'Path to a .h5 recording file')
    p_scans.add_argument('--stream', default = 'lidar_scan', help = 'Scan stream name')
    p_scans.add_argument('--bins',   default = 180, type = int, help = 'Angular bin count')
    p_scans.add_argument('--worst',  default = 10,  type = int, help = 'Worst scans to list')

    p_dump = sub.add_parser('dump', help = 'Print decoded events of one stream')
    p_dump.add_argument('path',   help = 'Path to a .h5 recording file')
    p_dump.add_argument('stream', help = 'Stream name (see summary)')
    p_dump.add_argument('--start', default = 0,  type = int, help = 'First event index')
    p_dump.add_argument('--count', default = 20, type = int, help = 'Number of events')

    args = parser.parse_args(argv)
    if args.command == 'summary':
        print_summary(args.path)
    elif args.command == 'scans':
        print_scan_stats(args.path, stream = args.stream,
                         bin_count = args.bins, worst = args.worst)
    elif args.command == 'dump':
        print_dump(args.path, args.stream, start = args.start, count = args.count)


if __name__ == '__main__':
    main()
