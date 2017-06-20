#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""normalize_feat.py: For feat input format, normalize each feature to follow normal distribution.
                        The statistics can be collected on separate data, i.e. normally
                        we would collect statistics on some training (sub) data, and then
                        apply the same transformation to training, dev, test, etc.

"""

__author__      = "Matthias Sperber"

def usage():
    print """usage: normalize_feat.py [options] stat-file-in.feat data-file-in.feat
    -h --help: print this Help message
    -m --max-stat-data n: collect statistics only over the first n data items (default: n=10000)
"""

import sys
import re
import getopt
import numpy as np

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
class ModuleTest(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            optlist, args = getopt.getopt(argv[1:], 'hm:', ['help', 'max-stat-data='])
        except getopt.GetoptError, msg:
            raise Usage(msg)
        maxStatData = 10000
        for o, a in optlist:
            if o in ["-h", "--help"]:
                usage()
                exit(2)
            if o in ["-m", "--max-stat-data"]:
                maxStatData = int(a)
                
        if len(args) not in [2]:
            raise Usage("must contain two non-optional parameters")
        statInFileName = args[0]
        dataInFileName = args[1]

        ###########################
        ## MAIN PROGRAM ###########
        ###########################

        featVecs = []
        statDataCount = 0
        for line in open(statInFileName):
            for featStr in line.split(";"):
                curFeatVec = np.asarray([float(v) for v in featStr.split()])
                assert len(featVecs)==0 or len(featVecs[-1]) == len(curFeatVec)
                statDataCount += 1
                featVecs.append(curFeatVec)
                if statDataCount >= maxStatData:
                    break
            if statDataCount > maxStatData:
                break

        mean, var = np.mean(featVecs, axis=0), np.var(featVecs, axis=0)

        for line in open(dataInFileName):
            featVecs = []
            for featStr in line.split(";"):
                curFeatVec = np.asarray([float(v) for v in featStr.split()])
                curFeatVec = (curFeatVec - mean ) / np.sqrt(var)
                featVecs.append(" ".join([str(v) for v in curFeatVec]))
            print ";".join(featVecs)
        ###########################
        ###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
