class RaftConfig:
    def __init__(self, path, patch_size=256, border=32, tissue='Brain', overlap=0.125):
        self.path = path
        self.patch_size = patch_size
        self.border = border
        self.tissue = tissue
        self.small = False
        self.model = 'RAFT/models/raft-things.pth'
        self.overlap = overlap
        self.mixed_precision = False
        self.alternate_corr = False
        self.occlusion = False

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

