from collections import defaultdict
import numpy as np
from typing import List, Callable


def ensemble_predict(models_predictions: List[dict],
                     models_slice_scaling_params: List[tuple],
                     models_calibrations_funcs: List[Callable],
                     volume_metadata: dict,
                     recall_threshold) -> dict:
    """
    This function combines the predictions of multiple models into a final single scalar per exam.
    There are different model types - image classifier, image detector and clinical data regressor

    :param models_predictions: Each element in models_predictions is expected to be one of:
        1. A python dict mapping a volume_identifier to a numpy array of scalars which are the slice scores
        This represents the models that are based on a slice-classifier.

        2. A python dict mapping a volume_identifier to a numpy array of numpy arrays representing the predicted
        bboxes and their respective scores.

            A bbox is represented as an array of 5 elements, the last of which is the malignancy score

            It is assumed that every slice has at least on bbox.

        This represents the models that are based on slice-level-detectors

        3. A python dict mapping (exam_identifier) to a scalar
        This represented the models that are based on clinical information (such as age, bmi, etc.)

        4. A python dict mapping volume_identifier to a scalar that is the volume score
        This represents the models that are volume classifiers.

    :param models_slice_scaling_params: list of min and max values per model

    :param models_calibrations_funcs: list of calibration functions per model

    :param volume_metadata: maps a volume_identifier to (exam_identifier, side, view)

    :return: a python dict which maps from exam identifier to a tuple (malignancy_score:single scalar, recall:Boolean)
    """

    #
    # Process detection predictions, get a single score per slice, thus making this a slice level score
    #
    for i, model_prediction in enumerate(models_predictions):
        # Take the max bounding box score for every slice
        if is_slice_level_detector(model_prediction):
            for volume_identifier in model_prediction:
                models_predictions[i][volume_identifier] = [np.max(bbox[:, 5])
                                                            for bbox in model_prediction[volume_identifier]]

    #
    # Collect slice level scores
    #
    volume_slice_scores = defaultdict(list)

    for i, model_prediction in enumerate(models_predictions):
        # collect slice scores from slice classifiers and normalize them
        if is_slice_classifier(model_prediction):
            min_val = models_slice_scaling_params[i][0]
            max_val = models_slice_scaling_params[i][1]
            for volume_identifier in model_prediction:
                volume_slice_scores[volume_identifier].append((model_prediction[volume_identifier] - min_val) / (max_val - min_val))

    #
    # volume_slice_scores is now a dict mappings from a volume identifier to a list of lists of normalized slice scores
    #
    # compute the representative score for each volume for each model
    #
    per_volume_scores = {}
    for volume_identifier in volume_slice_scores:
        per_volume_scores[volume_identifier] = slice_based_scores(np.stack(volume_slice_scores[volume_identifier]))

    # replace the per-slice classifications with volume classifications
    for i in range(len(models_predictions)):
        if is_slice_classifier(models_predictions[i]):
            for volume_identifier in per_volume_scores:
                models_predictions[i][volume_identifier] = per_volume_scores[volume_identifier][i]

    # replace per-volume classifications with exam classifications
    models_exam_predictions = get_per_exam_predictions(models_predictions, volume_metadata)

    #
    # calibrate models' exam predictions  (per model Plat's scaling)
    #
    for i in range(len(models_exam_predictions)):
        model_calibrations_func = models_calibrations_funcs[i]
        for exam_identifier, exam_score in models_exam_predictions[i].items():
            models_exam_predictions[i][exam_identifier] = model_calibrations_func(exam_score)

    # compute final prediction for each exam by taking the average exam (calibrated) score over all models
    exam_identifiers = models_exam_predictions[0].keys()
    exam_final_predictions = {}
    for exam_identifier in exam_identifiers:
        exam_final_predictions[exam_identifier] = [(np.mean(model_exam_prediction[exam_identifier]),
                                                    np.mean(model_exam_prediction[exam_identifier]) > recall_threshold)
                                                   for model_exam_prediction in models_exam_predictions]

    return exam_final_predictions


def is_slice_level_detector(model_prediction: dict) -> bool:
    """
    Is this a prediction of a slice level detector?

    :param model_prediction: a mapping from volume identifiers to classification results
    :return: Boolean
    """
    volume = list(model_prediction.keys())[0]

    return isinstance(model_prediction[volume], np.ndarray) and type(model_prediction[volume][0]) is np.ndarray


def is_slice_classifier(model_prediction: dict) -> bool:
    """
    Is this a prediction of a slice classifier?

    :param model_prediction: a mapping from volume identifiers to classification results
    :return: Boolean
    """
    volume = list(model_prediction.keys())[0]

    return isinstance(model_prediction[volume], np.ndarray) and type(model_prediction[volume][0]) is not np.ndarray


def slice_based_scores(slice_scores: np.ndarray) -> dict:
    """
    Compute volume scores using the score of the slice that is strongest over all models

    :param slice_scores: slice scores[model][slice]
    :return: volume[model] scores
    """
    best_slice = np.argmax(np.sum(slice_scores, axis=0))

    return slice_scores[:, best_slice]


def get_per_exam_predictions(models_predictions: List[dict], volume_metadata: dict) -> List[dict]:
    """
    Replace volume based predictions with slice based ones

    :param models_predictions: a mapping from identifiers to classification results
           At this point, only volume and exam modifiers are included in models_predictions
    :param volume_metadata: maps a volume_identifier to (exam_identifier, side, view)

    :return: a list of mappings from exam identifiers to exam malignancy scores
    """
    for i_model, model_predictions in enumerate(models_predictions):
        if is_volume_classifier(model_predictions, volume_metadata):
            # map each view_identifier to its predicted score. view_identifier = (exam_identifier, side, view)
            model_view_predictions = get_view_predictions(model_predictions, volume_metadata)

            # map each side_identifier to its predicted score. side_identifier = (exam_identifier, side)
            model_side_predictions = get_side_predictions(model_view_predictions)

            # map each exams_identifier to its predicted score.
            model_exam_predictions = get_exam_predictions(model_side_predictions)
            models_predictions[i_model] = model_exam_predictions

    return models_predictions


def is_volume_classifier(model_prediction: dict, volume_metadata: dict) -> bool:
    """
    Is this a prediction of a volume classifier?

    :param model_prediction: a mapping from  identifiers to classification results
    :param volume_metadata: maps a volume_identifier to (exam_identifier, side, view)
    :return: Boolean
    """
    key = list(model_prediction.keys())[0]

    return key in volume_metadata


def get_view_predictions(model_volume_predictions: dict, volume_metadata: dict) -> dict:
    """
    Returns the model's prediction by view by taking the maximum prediction of all volumes of the same view
    :param model_volume_predictions:
    :param volume_metadata: maps a volume_identifier to (exam_identifier, side, view)
    :return: a dictionary mapping view identifier, i.e. (exam_identifier, side, view), to the maximal score for the view
    """

    # group volume scores per view
    view_scores_dict = defaultdict(list)
    for volume_identifier, view_identifier in volume_metadata.items():
        view_scores_dict[view_identifier].append(model_volume_predictions[volume_identifier])

    # aggreate the group of each view by taking the maximum
    model_view_predictions = {}
    for view_identifier, view_scores in view_scores_dict.items():
        model_view_predictions[view_identifier] = np.nanmax(view_scores)
    return model_view_predictions


def get_side_predictions(model_view_predictions: dict) -> dict:
    """
    Returns the model's prediction by side by taking the average prediction of all view scores of that side
    :param model_view_predictions:
    :return: a dictionary mapping side identifier, i.e. (exam_identifier, side), to the average score for the side
    """
    # group view scores per side
    side_scores_dict = defaultdict(list)
    for (exam_identifier, side, _), view_score in model_view_predictions.items():
        side_identifier = (exam_identifier, side)
        side_scores_dict[side_identifier].append(view_score)

    # aggreate the group of each side by taking the average
    model_side_predictions = {}
    for side_identifier, side_scores in side_scores_dict.items():
        model_side_predictions[side_identifier] = np.nanmean(side_scores)
    return model_side_predictions


def get_exam_predictions(model_side_predictions: dict) -> dict:
    """
    Returns the model's prediction for the exam by taking the maximal prediction for the two sides
    :param model_side_predictions:
    :return: a dictionary mapping exam identifier to the max score of the two sides
    """
    # group view scores per exam
    exams_scores_dict = defaultdict(list)
    for (exam_identifier, _), side_score in model_side_predictions.items():
        exams_scores_dict[exam_identifier].append(side_score)

    # aggreate the group of each side by taking the average
    model_exam_predictions = {}
    for exam_identifier, exam_scores in exams_scores_dict.items():
        model_exam_predictions[exam_identifier] = np.nanmax(exam_scores)
    return model_exam_predictions
