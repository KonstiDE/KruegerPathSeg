class Connection:
    def __init__(self, start, end, path, cost, true_angle_diff, angle_infused_cost):

        self.start = start
        self.end = end
        self.path = path
        self.cost = cost
        self.true_angle_diff = true_angle_diff
        self.angle_infused_cost = angle_infused_cost
