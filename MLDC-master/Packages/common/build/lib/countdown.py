import sys, time

class countdown(object):
    def __init__(self,totalitems,interval=10000):
        self.t0    = time.time()
        self.total = totalitems
        self.int   = interval

        self.items = 0
        self.tlast = self.t0
        self.ilast = 0

        self.longest = 0

    def pad(self,outstring):
        if len(outstring) < self.longest:
            return outstring + " " * (self.longest - len(outstring))
        else:
            self.longest = len(outstring)
            return outstring

    def status(self,local=False,status=None):
        self.items = self.items + 1

        if self.items % self.int != 0:
            return

        t = time.time()

        speed      = float(self.items) / (t - self.t0)
        localspeed = float(self.items - self.ilast) / (t - self.tlast)

        if self.total != 0:
            eta      = (self.total - self.items) / speed
            localeta = (self.total - self.items) / localspeed
        else:
            eta, localeta = 0, 0

        self.tlast = t
        self.ilast = self.items

        print(self.pad("\r%d/%d done, %d s elapsed (%f/s), ETA %d s%s" % (self.items,self.total,t - self.t0,
                                                                          localspeed if local else speed,
                                                                          localeta if local else eta,
                                                                          ", " + status if status else "")   ), end=' ')
        sys.stdout.flush()

    def end(self,status=None):
        t = time.time()

        speed = self.items / (t - self.t0)

        print(self.pad("\r%d finished, %d s elapsed (%d/s)%s" % (self.items,t - self.t0,speed,", " + status if status else "")))
        sys.stdout.flush()
