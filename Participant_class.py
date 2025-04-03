class individual_participant_class:
    def __init__(self, name):
        self.name = name
        self.events = {}
        self.raw_intensity = None
        self.raw_od = None
        self.raw_haemo_unfiltered = None
        self.raw_haemo = None
        self.epochs = None
    
    def get_name(self):
        return self.name