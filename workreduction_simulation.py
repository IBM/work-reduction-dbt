

def simulate_work_reduction(
    ai_predictions: dict,
    radiologist_predictions: dict,
    ai_threshold: float,
) -> dict:
    '''
    A simple but effective combination of AI and radiologists in "work reduction" workflow scenario.
    AI examines all cases first, assigns part of the cases as "automatic non-recall", and the rest as requiring 
    human radiologist interpertation.
    In our study this configuration was shown to:
    1. Reduce ~40% of the radiologist workload
    2. Improve recall-rate significantly, avoiding 25% of the unnecessary patients recall compared to the standalone radiologist scenario
    3. Maintain non-inferior sensitivity to the standalone-radiologist scenario

    :param ai_predictions: a dict that maps from exam identifier to a scalar, representing the AI prediction
    prediction scores are assumed to be within the range [0.0,1.0]

    :param radiologist_predictions: a dict that maps from exam identifier to a scalar, representing the radiologist prediction
    prediction scores are assumed to be within the range [0.0,1.0]
    
    :param ai_threshold: a predefined threshold for the AI

    :result: a dict mapping from exam identifier to a scalar, simulating the joint AI and radiologist end-to-end system
    '''

    #verify we have predictions for all exams, by both AI and human radiologists
    assert set(ai_predictions.keys()) == set(radiologist_predictions.keys())

    answer = {}

    for exam_id, ai_pred_score in ai_predictions.items():
        if ai_pred_score<ai_threshold:
            answer[exam_id] = 0.0 #automatic non-recall
        else:
            answer[exam_id] = radiologist_predictions[exam_id] #use radiologist prediction

    return answer
