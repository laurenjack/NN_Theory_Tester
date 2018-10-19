import prediction_analytics as pa
import reporter as rep


def create_reporter(data_set):
    """Create a reporter to show/write the outcome of experiments run on various models.
    """
    prediction_analytics = pa.PredictionAnalytics(data_set.num_class)
    return rep.Reporter(data_set.num_class, prediction_analytics)
