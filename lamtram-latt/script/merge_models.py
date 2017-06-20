#!/usr/bin/env python

"""merge_models.py: Merge 2 model files by selecting some submodels from the 1st model, other submodels from the 2nd.
    Can be used to pretrain only parts of a model, etc.
    
"""

__author__ = "Matthias Sperber"
__date__   = "1/13/2017"

def usage():
    print """usage: merge_models.py [options] merge_desc model1 model2
    -h --help: print this Help message
    
    merge_desc looks like "dict_v001:1 dict_v001:2 extatt_005:1 linenc_004:1 linenc_004:1 nlm_005:2",
    describing the from where to take which submodel
"""

import getopt
import sys

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
class ModuleTest(Exception):
    def __init__(self, msg):
        self.msg = msg


def read_model(fname, sorted_keys):
    sorted_keys = list(sorted_keys) # make a copy
    fIn = open(fname)
    cur_lines = []
    model_contents = []
    cur_key = None
    next_key = sorted_keys.pop(0)
    for line in fIn:
        if next_key is not None and next_key in line:
            if cur_key is not None:
                model_contents.append(cur_lines)
                cur_lines = []
            cur_key = next_key
            try:
                next_key = sorted_keys.pop(0)
            except IndexError:
                next_key = None
        cur_lines.append(line.strip())
    if len(cur_lines)>0:
        model_contents.append(cur_lines)
    fIn.close()
    return model_contents


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
        EXPECTED_NUM_PARAMS = 3
        if len(args)!=EXPECTED_NUM_PARAMS:
            raise Usage("must contain %s non-optional parameter(s) (received %s)" % (EXPECTED_NUM_PARAMS, len(args)))
        merge_desc = args[0]
        inFileName1 = args[1]
        inFileName2 = args[2]
        ###########################
        ## MAIN PROGRAM ###########
        ###########################
        
        sorted_keys = [item.split(":")[0] for item in merge_desc.split()]
        
        model1 = read_model(inFileName1, sorted_keys)
                
        model2 = read_model(inFileName2, sorted_keys)

        
        for i, item in enumerate(merge_desc.split()):
            key, no = item.split(":")
            no = int(no)
            if no==1:
                submodel = model1[i]
            else:
                submodel = model2[i]
            for line in submodel:
                print line
            
        
        ###########################
        ###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
