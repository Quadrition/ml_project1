class Cluster:

    def __init__(self, center):
        self.center = center
        self.data = []

    def recalculate_center(self):
        new_center = [0 for i in range(len(self.center))]
        for d in self.data:
            for i in range(len(d)):
                new_center[i] += d[i]

        n = len(self.data)
        if n != 0:
            self.center = [x/n for x in new_center]
