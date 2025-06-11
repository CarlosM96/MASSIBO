from oei import *
# from daphne_ip import DAPHNE_IP
import time

#thing = OEI("192.168.133.12")


thing = OEI("192.168.0.200")


# number of times we should bitslip before giving up
BSRETRY = 100

# define the minimum bit width (in delay taps)
MINWIDTH = 13

print("DAPHNE firmware version %0X" % thing.read(0x9000,1)[2])

print("Resetting IDELAY and ISERDES...")
thing.write(0x2001, [0xdeadbeef])
#thing.write(0x2001, [1234])
print("pass")

# looking at the frame markers find the bit edges 
# and determine the ideal delay tap value for each AFE

for afe in range(5):
    print("AFE%d :" % afe, end=" ")
    for r in range(BSRETRY):
        x = []
        w = 0
        c = 0
        es = ""
        for dv in range(32): # scan all 32 delay values
            thing.write(0x4000+afe, [dv]) # write delay value
            thing.write(0x2000, [1234]) # trigger
            x.append(thing.read((0x40000000+(0x100000*afe)+(0x80000)),5)[3])
        for i in range(32): # x contains 32 samples each with different delay
            if (x[i]==0x3F80): # test for the proper frame marker
                w = w + 1
                c = c + i
                es = es + "*"
            else:
                es = es + "."
        if (x[0]==0x3F80 and x[31]==0x3F80): # we want to avoid double hump alignment
            continue
        if (w >= MINWIDTH):
            print("[%s] ALIGNED OK! bit width %d, using IDELAY tap %d" % (es, w, int(c/w)))
            thing.write(0x4000+afe, [int(c/w)]) # program idelay with optimal value
            break
        else:
            thing.write((0x3000+(0x10*afe)), [0,1,2,3,4,5,6,7,8]) # bitslip all AFE channels
        if (r==BSRETRY-1):
            print("FAILED!")

thing.close()
