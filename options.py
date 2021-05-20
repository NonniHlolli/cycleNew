class Options():

    def __init__(self,epoch = 0, n_epochs = 50, batchSize = 1, dataroot = './apple2orange', lr = 0.0002, decay_epoch = 100, size = 256, input_nc = 3, output_nc = 3, cuda = False, n_cpu = 8):
        # Starting epoch
        self.epoch = epoch
        # Number of epochs
        self.n_epochs = n_epochs
        # batch size
        self.batchSize = batchSize
        # directory of data
        self.dataroot = dataroot
        # initial learning rate
        self.lr = lr
        # how many epochs before linear decay starts
        self.decay_epoch = decay_epoch
        # size of data
        self.size = size
        # input channels
        self.input_nc = input_nc
        # output channels
        self.output_nc = output_nc
        # use GPU
        self.cuda = cuda
        # no. of cpu
        self.n_cpu = n_cpu


class TestOptions():

    def __init__(batchSize = 1, dataroot = './apple2orange', size = 256, input_nc = 3, output_nc = 3, cuda = False, n_cpu = 8, generator_A2B = 'output/netG_A2B.pth',  generator_B2A = 'output/netG_B2A.pth'):
        self.batchSize = batchSize
        # size of data
        self.size = size
        # input channels
        self.input_nc = input_nc
        # output channels
        self.output_nc = output_nc
        # use GPU
        self.cuda = cuda
        # no. of cpu
        self.n_cpu = n_cpu
        # directory of data
        self.dataroot = dataroot
        # directory of generator A to B
        self.generator_A2B = generator_A2B
        # directory of generator A to B
        self. generator_B2A = generator_B2A
