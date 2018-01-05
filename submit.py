import os
import sys
import cv2
import argparse


def progressbar(it, prefix="", size=60):
    count = len(it)

    def _show(_i):
        x = int(size*_i/count)
        sys.stdout.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), _i, count))
        sys.stdout.flush()

    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i+1)
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_length_encoding(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def save_submission_file():
    result_data_path = os.path.join('result')
    images = os.listdir(result_data_path)
    total = len(images)

    img_ids = []
    img_rles = []
    for i in progressbar(range(total), "Computing: ", 40):
        img_result = cv2.imread('./result/{}.png'.format(i+1), cv2.IMREAD_GRAYSCALE)
        img_id = i+1
        img_ids.append(img_id)
        img_rle = run_length_encoding(img_result)
        img_rles.append(img_rle)

    first_row = 'img,pixels'
    file_name = 'val_result_ver' + str(args.ver) + '.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(img_ids[i]) + ',' + img_rles[i]
            f.write(s + '\n')
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver', type=float, help='version of program')
    args = parser.parse_args()
    if args.ver:
        print('Version: {}'.format(args.ver))
        save_submission_file()
    else:
        parser.error('--ver argument is required')
