from data_analysis.glm_analysis_clean import run_glm_analysis

class GLM_class:
    def __init__(self, name, HRF_model, drift_model):
        self.name = name
        self.HRF_model = HRF_model
        self.drift_model = drift_model
        self.has_run = False
        self.results = None
    
    def getName(self):
        return self.name
    
    def getHasRun(self):
        return self.has_run
    
    def getHRFModel(self):
        return self.HRF_model
    
    def getDriftModel(self):
        return self.drift_model
    
    def update_parameters(self, params):
        self.name = params.get("GLMName", self.name)
        self.HRF_model = params.get("HRF_model", self.HRF_model)
        self.drift_model = params.get("drift_model", self.drift_model)
    
    def runGLM(self, all_participants, current_loader, number_of_subjects):
        results =  run_glm_analysis(all_participants, current_loader, self.drift_model, self.HRF_model, number_of_subjects)
        self.results = results
    
    def setHasRun(self, status):
        self.has_run = status
    
    def getResults(self):
        return self.results