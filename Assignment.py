class Assignment:
    def __init__(self, g1_id, g2_id, confidence):
        self.g1_id = g1_id
        self.g2_id = g2_id
        self.con = confidence

    def __lt__(self, other):
        if self.con >= other.con:
            return False
        return True

    def __str__(self):
        return str(f'g1: {self.g1_id} <----> g2: {self.g2_id} with {self.con:.8f} confidence')

    def __repr__(self):
        return str(self)

    def get_text_format(self):
        return str(f'{self.g1_id}, {self.g2_id}, {self.con}')
