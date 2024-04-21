


class PreferenceModel():
    def __init__(self) -> None:
        """
        params:
        training_steps : int
        learning_rate : float

        """
        pass


    def train(self):
        """
        Main training loop for the preference model
        Takes in two (s, a) pairs from opposing trajectories, timestep based preference learning
        Uses loss function from https://arxiv.org/pdf/2301.12842.pdf, Equation 5
        Training loop takes in a number of trajectories
        """
        