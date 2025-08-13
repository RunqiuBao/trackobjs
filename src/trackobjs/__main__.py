import argparse
import trackobjs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='path to the input images/, timestamps.txt, detections/.')
    args = parser.parse_args()

    trackobjs.track_objects(args.datadir)

if __name__ == "__main__":
    main()
    