import numpy as np

loreg = 0
hireg = 1
control = np.uint64(0)
inc = np.uint64(2**63)
pha = np.uint64(2**62)
sca = np.uint64(2**61)

list_of_commands = []


def printreg(r, h, d):
    global list_of_commands
    # print("{:010x} {:016x}".format(r * 16 + h * 8, d))
    list_of_commands.append([r * 2 + h, d])


def wrapphase(p):
    if p < 0:
        return p + 2**40
    else:
        return p


def setup(flist, plist, slist):
    global control
    global list_of_commands
    list_of_commands = []

    for f in flist:
        control = control ^ inc
        printreg(128, loreg, f | control)

    for s in range(9):
        for f in range(32):
            phase = plist[s][f][0]
            # if phase < 0:
            #     phase += 2**40
            control = control ^ pha
            printreg(128, loreg, phase | control)
            phase = plist[s][f][1]
            # if phase < 0:
            #     phase += 2**40
            control = control ^ pha
            printreg(128, loreg, phase | control)

    for s in range(8):
        for f in range(32):
            control = control ^ sca
            printreg(128, loreg, slist[s][f] | control)


def test_setup():
    def getf(i):
        return np.uint64(0x20000000 * (1 + 2 * f))

    flist = np.zeros(32, dtype=np.uint64)
    plist = np.zeros((9, 32, 2), dtype=np.uint64)
    slist = np.zeros((8, 32), dtype=np.uint64)

    for f in range(32):
        flist[f] = getf(f)

    for s in range(9):
        for f in range(32):
            plist[s][f][0] = wrapphase(0xC000000000 + s * getf(f) * 100 -
                                       f * getf(f))
            plist[s][f][1] = wrapphase(0xC000000000 + s * getf(f) * 100 -
                                       f * getf(f) + getf(f) // 2)

    for s in range(8):
        for f in range(32):
            slist[s][f] = 10000 * (s + 1) // (2 * f + 1)

    setup(flist, plist, slist)


if __name__ == "__main__":
    test_setup()
