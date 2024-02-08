class EMDiffuseConfig():

    def __init__(self, config, path, phase, batch_size, gpu='0', z_times=None, port='21012', mean=2):
        self.path = path
        self.config = config
        self.phase = phase
        self.batch_size = batch_size
        self.gpu = gpu
        self.debug = False
        self.z_times = z_times
        self.port = port
        self.mean = mean

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
