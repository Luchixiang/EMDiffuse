class EMDiffuseConfig():

    def __init__(self, config, path, phase, batch_size, lr=5e-5, resume=None, gpu='0', subsample=None, port='21012', mean=2, step=None):
        self.path = path
        self.config = config
        self.phase = phase
        self.batch = batch_size
        self.gpu = gpu
        self.debug = False
        self.z_times = subsample
        self.port = port
        self.resume = resume
        self.mean = mean
        self.lr = lr
        self.step=step

    def __getattr__(self, item):
        # This method is called when an attribute access is attempted.
        try:
            return self.__dict__[item]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        # This method allows setting attributes directly.
        self.__dict__[key] = value

    def __contains__(self, item):
        # This enables the use of 'in' to check for attribute existence.
        return item in self.__dict__
