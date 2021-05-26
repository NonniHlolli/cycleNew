class Options():

    def __init__(self,epoch = 0, n_epochs = 50,  dataroot = 'apple2orange/',  decay_epoch = 100,  cuda = False, n_cpu = 8):
        # Starting epoch
        self.epoch = epoch
        # Number of epochs
        self.n_epochs = n_epochs
        # directory of data
        self.dataroot = dataroot
        # how many epochs before linear decay starts
        self.decay_epoch = decay_epoch
        # use GPU
        self.cuda = cuda
        # no. of cpu
        self.n_cpu = n_cpu
