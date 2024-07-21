
from .BasicInference import BasicInference


class OneShot(BasicInference):

    def generate_one_shot_inferece(self,input_text):
        return self.generate_output([24],input_text)


class FewShot(BasicInference):

    def generate_one_shot_inferece(self, input_text):
        return self.generate_output([11,17,24], input_text)
