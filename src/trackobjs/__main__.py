import argparse
import trackobjs
import baodebug
import os.path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='path to the input images/, timestamps.txt, detections/.')
    args = parser.parse_args()

    baodebug.debugutils.ConfigureRootLogger("info")  # config logger format
    baodebug.debugutils.SetDebugPath(os.path.join(args.datadir, "baodebug/"))  # create debug folder here and set to os.environ["DEBUG_PATH"]

    trackobjs.track_objects(args.datadir)

if __name__ == "__main__":
    main()
    