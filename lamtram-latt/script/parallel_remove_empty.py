#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""parallel_remove_empty.py: Given a parallel corpus, removes lines that are empty on
                                either source- or target-side

"""

__author__      = "Matthias Sperber"

def usage():
    print """usage: parallel_remove_empty.py [options] in1 in2 out1 out2
    -h --help: print this Help message
"""

import sys
import re
import getopt

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
            optlist, args = getopt.getopt(argv[1:], 'h', ['help'])
        except getopt.GetoptError, msg:
            raise Usage(msg)
        for o, a in optlist:
            if o in ["-h", "--help"]:
                usage()
                exit(2)
                
        if len(args) not in [4]:
            raise Usage("must contain four non-optional parameters")
        in1FileName = args[0]
        in2FileName = args[1]
        out1FileName = args[2]
        out2FileName = args[3]

        ###########################
        ## MAIN PROGRAM ###########
        ###########################

        in2F = open(in2FileName)
        out1F = open(out1FileName, "w")
        out2F = open(out2FileName, "w")
        for line1 in open(in1FileName):
            line2 = in2F.readline()
            if len(line1.strip()) > 0 and len(line2.strip()) > 0:
                out1F.write(line1)
                out2F.write(line2)
        in2F.close()
        out1F.close()
        out2F.close()
        
        ###########################
        ###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
