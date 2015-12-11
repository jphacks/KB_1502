import graph_drawer
import numpy


class OmoroiData(object):
    def __init__(self):
        self.omoroi_sequence = []
        self.speed = 0
        self.pos = 0

        self.k = 0.2
        self.g = 0.5

        self.omoroi_max = 10.0
        self.omoroi_min = 0.0

        self.speed_max = 0.25
        self.speed_min = -0.25

        print "minnano warai"

    def _update_pos(self,speed):
        self.pos = speed + self.pos
        self.pos = min(max(self.pos,self.omoroi_min),self.omoroi_max)
        return self.pos

    def _update_speed(self,power):
        self.speed = power + self.speed
        self.speed = min(max(self.speed,self.speed_min),self.speed_max)
        return self.speed

    def update_omoroi_sequence(self,mean_of_smile):
        power = self.k * ( mean_of_smile - self.g)
        pos = self._update_pos( self._update_speed(power=power)) + numpy.random.normal(0,0.1)
        self.omoroi_sequence.append(pos)

    def get_subsequence(self, sequence, length):
        if len(sequence) < length:
            length = len(sequence)
        return sequence[-length:]
