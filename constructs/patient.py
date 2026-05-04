from enum import Enum
from typing import List

class PATIENT_EDGE_TYPES(Enum):
    AGE = "AGE"
    GENDER = "HAS_GENDER"
    HAS_CONDITION = "HAS_CONDITION"
    HAD_PAST_CONDITION = "HAD_PAST_CONDITION"
    TAKES_MEDICATION = "TAKES_MEDICATION"
    TOOK_PAST_MEDICATION = "TOOK_PAST_MEDICATION"
    IS_HEALTHY = "IS_HEALTHY"
    IS_PREGNANT = "IS_PREGNANT"

class Patient:
    def __init__(self, patient_graph):
        self.conditions = set()
        self.medications = set()
        self.age_months = None
        self.gender = None
        self.biological_sex = None 
        self.is_healthy = None
        self.is_pregnant = None
        self.parse_graph(patient_graph)
    
    def parse_graph(self, patient_graph):
        for edge in patient_graph:
            edge_type, edge_text = edge['type'], edge['target']
            
            match edge_type:
                case PATIENT_EDGE_TYPES.AGE.value:
                    self.age_months = edge_text
                case PATIENT_EDGE_TYPES.GENDER.value:
                    self.gender = edge_text
                    self.biological_sex = edge_text
                case PATIENT_EDGE_TYPES.IS_HEALTHY.value:
                    self.is_healthy = edge_text
                case PATIENT_EDGE_TYPES.IS_PREGNANT.value:
                    self.is_pregnant = edge_text
                case PATIENT_EDGE_TYPES.HAS_CONDITION.value:
                    self.conditions.add(edge_text)
                case PATIENT_EDGE_TYPES.HAD_PAST_CONDITION.value:
                    self.conditions.add(edge_text)
                case PATIENT_EDGE_TYPES.TAKES_MEDICATION.value:
                    self.medications.add(edge_text)
                case PATIENT_EDGE_TYPES.TOOK_PAST_MEDICATION.value:
                    self.medications.add(edge_text)
    
    def __str__(self):
        return f"""
                    Age (months): {self.age_months} \n
                    Gender: {self.gender} \n
                    Bio sex: {self.biological_sex} \n
                    is_healthy: {self.is_healthy} \n
                    is_pregnant: {self.is_pregnant} \n
                    conditions: {self.conditions} \n
                    medications: {self.medications} \n
                """
                
        