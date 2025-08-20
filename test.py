total_epochs = len(self.all_epochs[0].drop_log) * self.class_instance.number_of_participants
remaining_epochs = sum(len(ep) for ep in self.all_epochs) if self.all_epochs else 0
excluded_epochs = total_epochs - remaining_epochs 