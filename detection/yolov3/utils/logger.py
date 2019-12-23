import tensorboard


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tensorboard.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tensorboard.summary.Summary(value=[
            tensorboard.summary.Summary.Value(tag=tag, simple_value=value)
        ])
        self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        summary = tensorboard.summary.Summary(value=[
            tensorboard.summary.Summary.Value(tag=tag, simple_value=value)
            for tag, value in tag_value_pairs
        ])
        self.writer.add_summary(summary, step)
